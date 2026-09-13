"""FlyKing 라이브 허브 — 인스타 릴스 ⇄ 초파리 뇌 모델 ⇄ 3D 초파리 뷰어 중계.

실행:
  ~/flybrain/insta/.venv/bin/python live/hub.py                      # 실제 인스타, 뇌 판단, 좋아요는 가짜(뷰어에만 표시)
  ~/flybrain/insta/.venv/bin/python live/hub.py --like=real          # 실제로 좋아요까지 누름(사장님이 켤 때만)
  ~/flybrain/insta/.venv/bin/python live/hub.py --decider=random     # 비교용 무작위 규칙
  ~/flybrain/insta/.venv/bin/python live/hub.py --source=mock        # 인스타 없이 인공 자극(회색·루밍·줄무늬·깜빡임)으로 시험
그다음 브라우저로 http://127.0.0.1:8765/ 을 연다.

흐름(릴스 하나, --decider=brain):
  1. 릴스 정보를 읽고 영상 프레임을 뷰어로 계속 보낸다(연출용).
  2. 첫 --brain-seconds초(기본 1초) 프레임을 초당 --brain-fps장(기본 10) 모아 뇌 서버(brain/visual/brain_eval.py --serve)로 보낸다.
     뇌 서버는 겹눈 변환(eye.py) → 광수용체 입력 → Shiu 전뇌 모델 → 도파민 뉴런·거대섬유 발화율을 돌려준다.
  3. judge()가 문턱(brain/visual/calibration.json)과 비교해 판정한다.
       도주   : 거대섬유 50ms 최고 발화 ≥ 60Hz, 또는 PPL1(처벌) 지수가 문턱을 넘고 PAM(보상) 지수보다 큼 → 바로 넘김
       좋아요 : PAM(보상 도파민 뉴런) 평균 발화 ≥ 문턱 → 「초파리 뇌가 강하게 반응한 영상에 좋아요」 후 오래 봄
       보통   : 둘 다 아님 → 조금 보고 넘김
  4. 앞다리가 닿는 순간 좋아요·넘기기를 실행하고, 릴스마다 전체 수치를 ~/flybrain/insta/results/brain/ 에 JSON으로 남긴다.
  ⚠ 초파리는 영상 내용(사람·자막·웃김)을 모른다. 밝기·움직임에 대한 타고난 시각 반응일 뿐이고, 연결 세기가 고정이라 학습·취향이 없다.

프로토콜(WebSocket /ws): 허브→뷰어 텍스트 JSON {type: hello|reel|brain|act|liked|progress}, 영상 프레임은 바이너리 JPEG.
뷰어→허브 {type: "touch", id}.
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = HERE / "web"
sys.path.insert(0, str(ROOT / "insta"))
sys.path.insert(0, str(ROOT / "brain" / "visual"))
from common import IG, PROFILE_DIR, RESULTS_DIR  # noqa: E402
from reels import READ_JS  # noqa: E402

import eye  # noqa: E402
from capture_clips import RAW_JS, H as CAP_H, W as CAP_W  # noqa: E402

BRAIN_PY = Path.home() / "flybrain" / "shiu-brain-model" / ".venv" / "bin" / "python"
BRAIN_EVAL = ROOT / "brain" / "visual" / "brain_eval.py"
CALIB = ROOT / "brain" / "visual" / "calibration.json"
BRAIN_DIR = RESULTS_DIR / "brain"
NEUTRAL_WATCH_S = 5.0
BRAIN_SETTLE_S = 1.3        # 릴스로 넘어온 뒤(허브 next()가 이미 약 0.7초 기다림) 캡처 전 추가 대기 — 기준 릴스 캡처 시점(약 2초)과 맞춤
LIKE_EXTRA_WATCH_S = 5.0

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


def judge(res, calib):
    """뇌 결과 → 판정. 문턱·값·지수를 모두 담아 돌려준다(기록·뷰어 표시용).

    순서(앞이 우선):
      1. 도주   거대섬유 50ms 최고 ≥ 60Hz (von Reyn 2014; 문턱 embodied brain_body_bridge)          → 바로 넘김
      2. 처벌   PPL1 평균 ≥ 기준 릴스 90백분위 (Aso 2010·2012)                                         → 바로 넘김
      3. 보상   PAM 평균 ≥ 회색 대조군 평균+3SD (Liu 2012; Burke 2012)                                 → 좋아요
      4. 다가가기 oDN1·P9 평균 ≥ 기준 릴스 67백분위, 그리고 PPL1 ≤ 67백분위 (Bidaye 2020)            → 좋아요
      5. 그 밖                                                                                         → 조금 보고 넘김
    """
    pam = res["dopamine"]["PAM"]["mean_hz"]
    ppl1 = res["dopamine"]["PPL1"]["mean_hz"]
    gf = res["groups"]["GF"]["peak50ms_hz"]
    appr = round((res["groups"]["oDN1"]["mean_hz"] + res["groups"]["P9"]["mean_hz"]) / 2, 3)
    thr = (calib or {}).get("threshold", {})
    t_pam, t_gf = thr.get("PAM"), thr.get("GF_peak50ms_hz", 60.0)
    t_appr, t_ppl1_like, t_ppl1_avoid = thr.get("approach_like"), thr.get("PPL1_like_max"), thr.get("PPL1_avoid")
    idx = lambda v, t: round(v / t, 3) if t else None  # noqa: E731
    out = {
        "values": {"PAM_mean_hz": pam, "approach_hz": appr, "PPL1_mean_hz": ppl1, "GF_peak50ms_hz": gf},
        "thresholds": {"PAM_mean_hz": t_pam, "approach_hz": t_appr, "PPL1_like_max": t_ppl1_like,
                       "PPL1_mean_hz": t_ppl1_avoid, "GF_peak50ms_hz": t_gf},
        "threshold_source": (calib or {}).get("method", "보정 파일 없음 — brain/visual/calibrate.py 실행 필요"),
        "reward_index": idx(pam, t_pam), "approach_index": idx(appr, t_appr),
        "punish_index": idx(ppl1, t_ppl1_avoid), "escape_index": idx(gf, t_gf),
    }
    if not thr:
        out.update(verdict="neutral", reason="문턱 보정 파일이 없어 판정하지 않음 — brain/visual/calibrate.py 실행 필요")
    elif gf >= t_gf:
        out.update(verdict="avoid", reason=f"거대섬유(도약 도주) 최고 발화 {gf}Hz ≥ {t_gf}Hz → 바로 넘김")
    elif t_ppl1_avoid is not None and ppl1 >= t_ppl1_avoid and ppl1 > 0:
        out.update(verdict="avoid", reason=f"처벌 도파민 PPL1 {ppl1}Hz ≥ 기준 릴스 상위 10% {t_ppl1_avoid}Hz → 바로 넘김")
    elif t_pam and pam >= t_pam:
        out.update(verdict="like", reason=f"보상 도파민 PAM {pam}Hz ≥ 회색 화면 문턱 {t_pam}Hz → 초파리 뇌가 강하게 반응한 영상에 좋아요")
    elif t_appr and appr >= t_appr and (t_ppl1_like is None or ppl1 <= t_ppl1_like):
        out.update(verdict="like", reason=f"다가가기 명령 oDN1·P9 {appr}Hz ≥ 기준 릴스 상위 1/3 {t_appr}Hz, 처벌 PPL1은 낮음 "
                                          f"→ 초파리 뇌가 강하게 반응한 영상에 좋아요(보상 도파민 PAM {pam}Hz)")
    else:
        out.update(verdict="neutral", reason=f"다가가기 {appr}Hz < {t_appr}Hz 또는 처벌 PPL1이 높음, 보상 PAM {pam}Hz → 조금 보고 넘김")
    return out


class BrainClient:
    """뇌 모델 환경의 brain_eval.py --serve 를 자식 프로세스로 띄워 둔다(네트워크는 한 번만 만든다)."""

    def __init__(self):
        self.proc = None
        self.pending = {}
        self.ready = asyncio.Event()
        self.build_s = None

    async def start(self):
        (BRAIN_DIR / "tmp").mkdir(parents=True, exist_ok=True)
        errlog = open(BRAIN_DIR / "brain_server.log", "ab")
        self.proc = await asyncio.create_subprocess_exec(
            str(BRAIN_PY), str(BRAIN_EVAL), "--serve", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=errlog, limit=64 * 1024 * 1024)
        asyncio.create_task(self._reader())
        log("BRAIN_STARTING (네트워크 만드는 중 — 수십 초~몇 분)")

    async def _reader(self):
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                log("BRAIN_EXITED — ~/flybrain/insta/results/brain/brain_server.log 확인")
                for fut in self.pending.values():
                    if not fut.done():
                        fut.set_exception(RuntimeError("brain server exited"))
                return
            m = json.loads(line)
            if m.get("type") == "ready":
                self.build_s = m.get("build_s")
                self.ready.set()
                log("BRAIN_READY", self.build_s)
            else:
                fut = self.pending.pop(m.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(m)

    async def evaluate(self, frames, fps, key, timeout=300):
        path = BRAIN_DIR / "tmp" / f"{key}.npz"
        np.savez(path, frames=frames, fps=fps)
        fut = asyncio.get_running_loop().create_future()
        self.pending[key] = fut
        self.proc.stdin.write((json.dumps({"id": key, "npz": str(path)}) + "\n").encode())
        await self.proc.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            path.unlink(missing_ok=True)


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
        self.brain = BrainClient() if args.decider == "brain" else None
        self.calib = json.loads(CALIB.read_text()) if CALIB.exists() else None

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
        await ws.send_str(json.dumps({"type": "hello", "source": self.args.source, "like": self.args.like,
                                      "decider": self.args.decider}))
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

    def random_plan(self, info):
        """비교용 무작위 규칙(--decider=random). 볼 시간과 좋아요 시각(릴스 시작 기준 초)."""
        dur = finite(info.get("duration"), 10.0)
        watch = min(max(dur, 4.0), self.args.max_watch) * self.rng.uniform(0.45, 1.0)
        like_at = watch * self.rng.uniform(0.35, 0.65) if (not info.get("liked") and self.rng.random() < self.args.like_prob) else None
        return {"watch": watch, "like_at": like_at, "judge": {"verdict": "random", "reason": f"무작위 규칙(좋아요 확률 {self.args.like_prob})"}}

    async def brain_plan(self, src, info, rid, t_start):
        # 기준 릴스(capture_clips.py)는 넘긴 뒤 약 2초 뒤 첫 1초를 캡처했다. 허브도 같은 시점에 캡처해야 보정 문턱과 맞는다.
        # (2026-09-13: 바로 캡처하면 넘기는 화면 전환·영상 시작 밝아짐이 다가옴으로 잡혀 거대섬유 도주가 5개 중 4개로 과다)
        await asyncio.sleep(BRAIN_SETTLE_S)
        await self.send_all({"type": "brain", "id": rid, "status": "capturing"})
        frames = await src.capture(self.args.brain_seconds, self.args.brain_fps)
        await self.send_all({"type": "brain", "id": rid, "status": "computing"})
        key = hashlib.sha1(f"{rid}{time.time()}".encode()).hexdigest()[:12]
        t0 = time.monotonic()
        res = await self.brain.evaluate(frames, self.args.brain_fps, key)
        if res.get("type") != "result":
            raise RuntimeError(res.get("error", "brain error"))
        j = judge(res, self.calib)
        elapsed = time.monotonic() - t_start
        dur = finite(info.get("duration"), 10.0)
        if j["verdict"] == "avoid":
            watch, like_at = elapsed + 0.5, None
        elif j["verdict"] == "like" and not info.get("liked"):
            like_at = elapsed + 0.3
            watch = max(elapsed + LIKE_EXTRA_WATCH_S, min(dur, self.args.max_watch))
        else:
            watch, like_at = max(elapsed + 1.5, NEUTRAL_WATCH_S), None
        return {"watch": watch, "like_at": like_at, "judge": j, "brain": res, "brain_wall_s": round(time.monotonic() - t0, 2),
                "frames_captured": int(len(frames))}

    async def session(self, src):
        if self.brain:
            await self.brain.start()
            await self.brain.ready.wait()
            if not self.calib:
                log("WARN_NO_CALIBRATION — brain/visual/calibrate.py를 먼저 돌리면 좋아요 판정이 켜진다")
        n = 0
        while self.args.max_reels == 0 or n < self.args.max_reels:
            info = await src.read()
            if not info:
                await asyncio.sleep(0.5)
                continue
            n += 1
            self.seen += 1
            rid = info["url"]
            t_start = time.monotonic()
            self.reel = {"type": "reel", "id": rid, "author": info.get("author"), "caption": info.get("caption"),
                         "likeCount": info.get("likeCount"), "liked": bool(info.get("liked")),
                         "duration": finite(info.get("duration"), 0), "watch": None, "likeAt": None,
                         "seen": self.seen, "likedTotal": self.liked_total, "decider": self.args.decider}
            await self.send_all(self.reel)
            try:
                plan = await self.brain_plan(src, info, rid, t_start) if self.brain else self.random_plan(info)
            except Exception as e:  # noqa: BLE001
                log("BRAIN_FAIL", repr(e))
                plan = {"watch": NEUTRAL_WATCH_S, "like_at": None, "judge": {"verdict": "error", "reason": f"뇌 계산 실패: {e!r}"}}
            j = plan["judge"]
            self.reel.update(watch=round(plan["watch"], 1), likeAt=None if plan["like_at"] is None else round(plan["like_at"], 1))
            await self.send_all(self.reel)
            await self.send_all({"type": "brain", "id": rid, "status": "done", **{k: j.get(k) for k in
                                 ("verdict", "reason", "values", "thresholds", "reward_index", "approach_index",
                                  "punish_index", "escape_index")}})
            log("REEL", json.dumps({"id": rid, "author": info.get("author"), "verdict": j.get("verdict"),
                                    "watch": self.reel["watch"], "likeAt": self.reel["likeAt"],
                                    "values": j.get("values"), "brain_wall_s": plan.get("brain_wall_s")}, ensure_ascii=False))

            liked_real = None
            if plan["like_at"] is not None:
                await asyncio.sleep(max(0.0, t_start + plan["like_at"] - time.monotonic()))
                touched = await self.request_action("like", rid)
                ok = await src.like() if self.args.like == "real" else True
                liked_real = ok if self.args.like == "real" else False
                self.liked_total += int(ok)
                await self.send_all({"type": "liked", "id": rid, "liked": ok, "fake": self.args.like != "real",
                                     "likedTotal": self.liked_total})
                log("LIKE", self.args.like, ok, "touched" if touched else "no-touch")
            await asyncio.sleep(max(0.0, t_start + plan["watch"] - time.monotonic()))
            await self.request_action("swipe", rid)
            watched = time.monotonic() - t_start
            moved = await src.next()
            log("SWIPE", moved)
            self.record(info, plan, watched, liked_real)
        log("SESSION_DONE", self.seen)

    def record(self, info, plan, watched, liked_real):
        """릴스마다 판정 근거를 남긴다: decisions.jsonl(한 줄 요약) + reels/<시각>.json(전체 수치)."""
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)
        (BRAIN_DIR / "reels").mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        j = plan["judge"]
        full = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "decider": self.args.decider, "like_mode": self.args.like,
            "reel": {k: info.get(k) for k in ("url", "author", "caption", "duration", "likeCount")},
            "capture": {"seconds": self.args.brain_seconds, "fps": self.args.brain_fps, "frames": plan.get("frames_captured")},
            "judge": j, "watched_s": round(watched, 2), "like_at_s": plan["like_at"], "liked_on_instagram": liked_real,
            "brain_wall_s": plan.get("brain_wall_s"),
            "readout_sources": {"approach(oDN1·P9)": "Bidaye et al. 2020 Neuron 108:469",
                                "PAM": "Liu et al. 2012 Nature 488:512; Burke et al. 2012 Nature 492:433",
                                "PPL1": "Aso et al. 2010 Curr Biol 20:1445; Aso et al. 2012 PLoS Genet 8:e1002768",
                                "GF": "von Reyn et al. 2014 Nat Neurosci 17:962; 문턱 embodied brain_body_bridge 0.3×200Hz"},
            "brain": plan.get("brain"),
        }
        path = BRAIN_DIR / "reels" / f"{stamp}.json"
        path.write_text(json.dumps(full, ensure_ascii=False, indent=1))
        line = {"time": full["time"], "url": info.get("url"), "author": info.get("author"), "verdict": j.get("verdict"),
                "reason": j.get("reason"), "values": j.get("values"), "thresholds": j.get("thresholds"),
                "reward_index": j.get("reward_index"), "punish_index": j.get("punish_index"), "escape_index": j.get("escape_index"),
                "input": (plan.get("brain") or {}).get("input"), "watched_s": full["watched_s"], "file": str(path)}
        with open(BRAIN_DIR / "decisions.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


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

    async def capture(self, seconds, fps):
        frames = []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end or not frames:
            t0 = time.monotonic()
            try:
                r = await self.page.evaluate(RAW_JS, [CAP_W, CAP_H])
                if r:
                    frames.append(np.frombuffer(base64.b64decode(r["rgb"]), np.uint8).reshape(CAP_H, CAP_W, 3))
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(max(0.0, 1.0 / fps - (time.monotonic() - t0)))
            if time.monotonic() > t_end + 3:
                break
        want = int(round(seconds * fps))
        if not frames:
            frames = [np.zeros((CAP_H, CAP_W, 3), np.uint8)]
        while len(frames) < want:
            frames.append(frames[-1])
        return np.stack(frames[:want])

    async def read(self):
        """릴스 정보 읽기. 넘기는 중 페이지가 바뀌면 evaluate가 실패할 수 있어 None을 돌려 세션이 다시 읽게 한다.
        (2026-09-13 사용자 실행에서 릴스 13개 뒤 여기서 예외로 허브 전체가 멈춤.) 창이 닫혔으면 분명히 끝낸다."""
        try:
            return await self.page.evaluate(READ_JS)
        except Exception as e:  # noqa: BLE001
            if self.page.is_closed() or "closed" in str(e).lower():
                raise SystemExit("INSTA_WINDOW_CLOSED — 인스타 창이 닫혀 허브를 끝냄") from e
            log("READ_RETRY", type(e).__name__, str(e).splitlines()[0][:120])
            await asyncio.sleep(1.0)
            return None

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
    """인스타 없이 시험: 인공 자극(eye.synthetic)과 저장한 릴스 조각(results/clips)을 번갈아 보여 준다."""

    def __init__(self, hub):
        self.hub = hub
        self.items = [("synthetic", k) for k in ("gray", "loom", "bars", "flicker")]
        self.items += [("clip", p) for p in sorted((RESULTS_DIR / "clips").glob("clip_*.npz"))]
        self.i = 0
        self.liked = False
        self.cache = {}

    def frames(self, seconds=2.0, fps=10.0):
        kind, what = self.items[self.i % len(self.items)]
        key = (kind, str(what))
        if key not in self.cache:
            if kind == "synthetic":
                self.cache[key] = eye.synthetic(what, seconds, fps)
            else:
                self.cache[key] = np.load(what)["frames"]
        return self.cache[key]

    async def start(self, p=None):
        log("MOCK_ITEMS", [f"{k}:{Path(str(w)).name}" for k, w in self.items])
        asyncio.create_task(self.frame_loop())

    async def frame_loop(self):
        from PIL import Image
        k = 0
        while True:
            if self.hub.clients:
                fr = self.frames()
                buf = io.BytesIO()
                Image.fromarray(fr[k % len(fr)]).resize((432, 768)).save(buf, "JPEG", quality=80)
                await self.hub.send_frame(buf.getvalue())
                k += 1
            await asyncio.sleep(0.1)

    async def capture(self, seconds, fps):
        await asyncio.sleep(seconds)
        fr = self.frames()
        want = int(round(seconds * fps))
        reps = int(np.ceil(want / len(fr)))
        return np.concatenate([fr] * reps)[:want]

    async def read(self):
        kind, what = self.items[self.i % len(self.items)]
        name = what if kind == "synthetic" else Path(str(what)).stem
        return {"url": f"mock://{kind}/{name}/{self.i}", "author": f"시험:{name}", "caption": "인스타 없이 시험하는 자극입니다",
                "likeCount": "—", "liked": self.liked, "duration": 9.0}

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
    ap.add_argument("--decider", choices=["brain", "random"], default="brain")
    ap.add_argument("--like", choices=["fake", "real"], default="fake")
    ap.add_argument("--like-prob", type=float, default=0.35, help="--decider=random 전용")
    ap.add_argument("--brain-seconds", type=float, default=1.0)
    ap.add_argument("--brain-fps", type=float, default=10.0)
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
    log(f"HUB_READY http://127.0.0.1:{args.port}/ source={args.source} decider={args.decider} like={args.like}")

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
    if hub.brain and hub.brain.proc:
        hub.brain.proc.stdin.close()
    await asyncio.sleep(1.0)
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
