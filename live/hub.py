"""FlyKing 라이브 허브 — 인스타 릴스 ⇄ 3D 초파리 뷰어 중계.

실행:
  ~/flybrain/insta/.venv/bin/python live/hub.py                      # 실제 인스타, 좋아요는 가짜(뷰어에만 표시)
  ~/flybrain/insta/.venv/bin/python live/hub.py --like=real          # 실제로 좋아요까지 누름
  ~/flybrain/insta/.venv/bin/python live/hub.py --source=mock        # 인스타 없이 저장된 프레임으로 시험
그다음 브라우저로 http://127.0.0.1:8765/ 을 연다.

흐름(릴스 하나):
  1. 허브가 인스타 화면에서 릴스 정보(작성자·길이·좋아요 수)를 읽고 영상 프레임을 초당 fps장 뷰어로 보낸다.
  2. 판단 규칙이 몇 초 볼지, 좋아요를 누를지 정한다. ⚠ 지금은 임시 무작위 규칙 — 뇌 모델 연결 전 자리.
  3. 좋아요 시각이 되면 뷰어에 「like」 동작을 요청 → 초파리 앞다리가 하트에 닿는 순간 뷰어가 touch를 보냄
     → 그때 허브가 인스타에서 좋아요를 누른다(가짜 모드면 누르지 않고 뷰어에만 표시).
  4. 다 보면 「swipe」 요청 → 앞다리로 화면을 끌어올린 순간 touch → 허브가 다음 릴스로 넘긴다.
  뷰어가 안 붙어 있거나 8초 안에 touch가 없으면 허브가 그냥 진행한다.

프로토콜(WebSocket /ws): 허브→뷰어 텍스트 JSON {type: hello|reel|act|liked|progress}, 영상 프레임은 바이너리 JPEG.
뷰어→허브 {type: "touch", id}.
"""
import argparse
import asyncio
import base64
import json
import math
import random
import sys
import time
from pathlib import Path

from aiohttp import WSMsgType, web

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = HERE / "web"
sys.path.insert(0, str(ROOT / "insta"))
from common import IG, PROFILE_DIR, RESULTS_DIR  # noqa: E402
from reels import READ_JS  # noqa: E402

# 화면에서 가장 많이 보이는 video를 cover 맞춤으로 JPEG 추출(2026-09-13 확인: canvas 오염 없음, 1장 약 67ms).
GRAB_JS = r"""
([W, H, q]) => {
  const vw = innerWidth, vh = innerHeight;
  let v = null, best = 0;
  for (const el of document.querySelectorAll('video')) {
    const r = el.getBoundingClientRect();
    const a = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0)) * Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    if (a > best) { best = a; v = el; }
  }
  if (!v || v.readyState < 2 || !v.videoWidth) return null;
  const c = window.__flyCanvas || (window.__flyCanvas = document.createElement('canvas'));
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  const s = Math.max(W / v.videoWidth, H / v.videoHeight);
  const dw = v.videoWidth * s, dh = v.videoHeight * s;
  g.drawImage(v, (W - dw) / 2, (H - dh) / 2, dw, dh);
  return { jpg: c.toDataURL('image/jpeg', q).slice(23), t: v.currentTime, d: v.duration };
}
"""


def log(*a):
    print(*a, flush=True)


def finite(x, default):
    return x if isinstance(x, (int, float)) and math.isfinite(x) and x > 0 else default


class Hub:
    def __init__(self, args):
        self.args = args
        self.clients = set()
        self.pending = {}
        self.reel = None
        self.last_frame = None
        self.seen = 0
        self.liked_total = 0
        self.rng = random.Random(args.seed)

    async def send_all(self, msg):
        data = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:  # noqa: BLE001
                self.clients.discard(ws)

    async def send_frame(self, jpg):
        self.last_frame = jpg
        for ws in list(self.clients):
            try:
                await ws.send_bytes(jpg)
            except Exception:  # noqa: BLE001
                self.clients.discard(ws)

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        log("VIEWER_CONNECTED", len(self.clients))
        await ws.send_str(json.dumps({"type": "hello", "source": self.args.source, "like": self.args.like}))
        if self.reel:
            await ws.send_str(json.dumps(self.reel, ensure_ascii=False))
        if self.last_frame:
            await ws.send_bytes(self.last_frame)
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
            except ValueError:
                continue
            if m.get("type") == "touch":
                fut = self.pending.get(m.get("id"))
                if fut and not fut.done():
                    fut.set_result(m)
        self.clients.discard(ws)
        log("VIEWER_LEFT", len(self.clients))
        return ws

    async def request_action(self, action, reel_id, timeout=8.0):
        """뷰어에 동작을 요청하고 앞다리가 닿을 때까지 기다린다. 닿았으면 True."""
        if not self.clients:
            await asyncio.sleep(0.3)
            return False
        key = f"{action}:{reel_id}:{time.time():.3f}"
        fut = asyncio.get_running_loop().create_future()
        self.pending[key] = fut
        log("ACT", action, reel_id)
        await self.send_all({"type": "act", "action": action, "id": key})
        try:
            await asyncio.wait_for(fut, timeout)
            log("TOUCH", action)
            return True
        except asyncio.TimeoutError:
            log("TOUCH_TIMEOUT", action)
            return False
        finally:
            self.pending.pop(key, None)

    def decide(self, info):
        """⚠ 임시 판단 규칙(뇌 모델 연결 전). 볼 시간(초)과 좋아요 시각(초, 안 누르면 None)."""
        dur = finite(info.get("duration"), 10.0)
        watch = min(max(dur, 4.0), self.args.max_watch) * self.rng.uniform(0.45, 1.0)
        like_at = None
        if not info.get("liked") and self.rng.random() < self.args.like_prob:
            like_at = watch * self.rng.uniform(0.35, 0.65)
        return watch, like_at

    async def session(self, src):
        n = 0
        while self.args.max_reels == 0 or n < self.args.max_reels:
            info = await src.read()
            if not info:
                await asyncio.sleep(0.5)
                continue
            n += 1
            self.seen += 1
            rid = info["url"]
            watch, like_at = self.decide(info)
            self.reel = {"type": "reel", "id": rid, "author": info.get("author"), "caption": info.get("caption"),
                         "likeCount": info.get("likeCount"), "liked": bool(info.get("liked")),
                         "duration": finite(info.get("duration"), 0), "watch": round(watch, 1),
                         "likeAt": None if like_at is None else round(like_at, 1),
                         "seen": self.seen, "likedTotal": self.liked_total}
            await self.send_all(self.reel)
            log("REEL", json.dumps({k: self.reel[k] for k in ("id", "author", "duration", "watch", "likeAt")}, ensure_ascii=False))

            t0 = time.monotonic()
            like_done = like_at is None
            while time.monotonic() - t0 < watch:
                await asyncio.sleep(0.2)
                if not like_done and time.monotonic() - t0 >= like_at:
                    like_done = True
                    await self.request_action("like", rid)
                    ok = await src.like() if self.args.like == "real" else True
                    self.liked_total += int(ok)
                    await self.send_all({"type": "liked", "id": rid, "liked": ok, "fake": self.args.like != "real",
                                         "likedTotal": self.liked_total})
                    log("LIKE", "real" if self.args.like == "real" else "fake", ok)
            await self.request_action("swipe", rid)
            moved = await src.next()
            log("SWIPE", moved)
        log("SESSION_DONE", self.seen)


class InstaSource:
    def __init__(self, hub):
        self.hub = hub
        self.page = None

    async def start(self, p):
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR), headless=False, viewport=None, locale="ko-KR",
            ignore_default_args=["--enable-automation"], args=["--window-size=1280,800"])
        self.page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await self.page.goto(f"{IG}/reels/")
        cookies = {c["name"] for c in await ctx.cookies(IG)}
        if "sessionid" not in cookies:
            raise SystemExit("NOT_LOGGED_IN — insta/login.py로 먼저 로그인")
        await self.page.wait_for_selector("video", timeout=30000)
        await self.page.wait_for_timeout(2500)
        asyncio.create_task(self.grab_loop())

    async def grab_loop(self):
        a = self.hub.args
        period, last_progress = 1.0 / a.fps, 0.0
        while True:
            t0 = time.perf_counter()
            if self.hub.clients:
                try:
                    r = await self.page.evaluate(GRAB_JS, [432, 768, 0.72])
                    if r and r.get("jpg"):
                        await self.hub.send_frame(base64.b64decode(r["jpg"]))
                        if t0 - last_progress > 0.5:
                            last_progress = t0
                            await self.hub.send_all({"type": "progress", "t": r["t"], "d": finite(r["d"], 0)})
                except Exception:  # noqa: BLE001  (넘기는 중 페이지가 바뀌면 한두 번 실패한다)
                    pass
            await asyncio.sleep(max(0.0, period - (time.perf_counter() - t0)))

    async def read(self):
        return await self.page.evaluate(READ_JS)

    async def like(self):
        info = await self.read()
        if not info or info.get("liked") is None:
            return False
        if info["liked"]:
            return True
        await self.page.locator('[data-fly="like"]').first.click()
        await self.page.wait_for_timeout(800)
        after = await self.read()
        return bool(after and after.get("liked"))

    async def next(self):
        before = self.page.url
        await self.page.keyboard.press("ArrowDown")
        for _ in range(20):
            await asyncio.sleep(0.15)
            if self.page.url != before:
                break
        await self.page.wait_for_timeout(700)
        return self.page.url != before


class MockSource:
    """인스타 없이 시험: 전에 저장한 릴스 프레임을 돌려 가며 보여 준다."""

    def __init__(self, hub):
        self.hub = hub
        self.files = sorted((RESULTS_DIR / "check").glob("frame_*.png")) + sorted((RESULTS_DIR / "probe").glob("reel_*.png"))
        self.i = 0
        self.liked = False

    async def start(self, p=None):
        log("MOCK_FRAMES", len(self.files))
        asyncio.create_task(self.frame_loop())

    async def frame_loop(self):
        while True:
            if self.files and self.hub.clients:
                await self.hub.send_frame(self.files[self.i % len(self.files)].read_bytes())
            await asyncio.sleep(0.5)

    async def read(self):
        return {"url": f"mock://reel/{self.i}", "author": f"mock_reel_{self.i}", "caption": "인스타 없이 시험하는 가짜 릴스입니다",
                "likeCount": f"{3 + self.i}.{self.i}만", "liked": self.liked, "duration": 9.0}

    async def like(self):
        self.liked = True
        return True

    async def next(self):
        self.i += 1
        self.liked = False
        return True


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["insta", "mock"], default="insta")
    ap.add_argument("--like", choices=["fake", "real"], default="fake")
    ap.add_argument("--like-prob", type=float, default=0.35)
    ap.add_argument("--max-watch", type=float, default=14.0)
    ap.add_argument("--max-reels", type=int, default=0, help="0이면 끝없이")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    hub = Hub(args)
    app = web.Application()

    async def index(_):
        raise web.HTTPFound("/web/index.html")

    app.router.add_get("/", index)
    app.router.add_get("/ws", hub.ws_handler)
    app.router.add_static("/web", WEB)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    log(f"HUB_READY http://127.0.0.1:{args.port}/ source={args.source} like={args.like}")

    if args.source == "mock":
        src = MockSource(hub)
        await src.start()
        await hub.session(src)
    else:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            src = InstaSource(hub)
            await src.start(p)
            await hub.session(src)
    await asyncio.sleep(1.0)
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
