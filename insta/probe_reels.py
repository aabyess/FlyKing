"""릴스 화면 구조 탐색(읽기 전용). 좋아요는 누르지 않고, 보기·넘기기만 한다.

로그인(login.py)이 끝난 뒤 실행:
  ~/flybrain/insta/.venv/bin/python insta/probe_reels.py [릴스 개수=3]

결과: ~/flybrain/insta/results/probe/
  reel_N.json   — 주소·길이·작성자·캡션·주변 버튼 aria-label
  reel_N.png    — 초파리 눈에 넣을 영상 프레임(video 요소 캡처)
  view_N.png    — 창 전체 스크린샷
"""
import json
import sys
import time

from playwright.sync_api import sync_playwright

from common import IG, RESULTS_DIR, logged_in_user, open_fly_chrome

OUT = RESULTS_DIR / "probe"

# 화면 한가운데에 가장 많이 걸친 video와 그 릴스 칸의 정보를 모은다.
INSPECT_JS = r"""
() => {
  const vids = [...document.querySelectorAll('video')];
  const vh = innerHeight, vw = innerWidth;
  let best = null, bestArea = 0;
  for (const v of vids) {
    const r = v.getBoundingClientRect();
    const w = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0));
    const h = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    if (w * h > bestArea) { bestArea = w * h; best = v; }
  }
  if (!best) return { videos: vids.length, found: false };
  best.setAttribute('data-fly-current', '1');
  document.querySelectorAll('video[data-fly-current]').forEach(v => { if (v !== best) v.removeAttribute('data-fly-current'); });
  // 버튼이 들어 있을 만큼 큰 조상 칸까지 올라간다.
  let box = best;
  for (let i = 0; i < 12 && box.parentElement; i++) {
    box = box.parentElement;
    if (box.querySelectorAll('svg[aria-label]').length >= 3) break;
  }
  const labels = [...box.querySelectorAll('svg[aria-label]')].map(s => s.getAttribute('aria-label'));
  const links = [...box.querySelectorAll('a[href]')].map(a => ({ href: a.getAttribute('href'), text: a.innerText.trim().slice(0, 60) }));
  const texts = [...box.querySelectorAll('span[dir="auto"], h1, div[dir="auto"]')].map(e => e.innerText.trim()).filter(Boolean);
  return {
    videos: vids.length, found: true, url: location.href,
    duration: best.duration, currentTime: best.currentTime, paused: best.paused, muted: best.muted,
    size: [best.videoWidth, best.videoHeight],
    labels, links: links.slice(0, 20), texts: [...new Set(texts)].slice(0, 15),
  };
}
"""


def main(n=3):
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = open_fly_chrome(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(f"{IG}/reels/")
        if not logged_in_user(ctx):
            print("NOT_LOGGED_IN login.py 먼저 실행", flush=True)
            ctx.close()
            return 2
        page.wait_for_selector("video", timeout=30000)
        page.wait_for_timeout(3000)

        for i in range(n):
            info = page.evaluate(INSPECT_JS)
            info["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (OUT / f"reel_{i}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
            page.screenshot(path=str(OUT / f"view_{i}.png"))
            vid = page.locator("video[data-fly-current]")
            if vid.count():
                vid.first.screenshot(path=str(OUT / f"reel_{i}.png"))
            print(f"REEL {i} url={info.get('url')} dur={info.get('duration')} labels={info.get('labels')}", flush=True)

            # 다음 릴스로 넘기기: 키보드 먼저, 주소가 안 바뀌면 휠.
            before = page.url
            page.keyboard.press("ArrowDown")
            page.wait_for_timeout(2500)
            if page.url == before:
                page.mouse.move(260, 470)
                page.mouse.wheel(0, 900)
                page.wait_for_timeout(2500)
            print(f"NEXT moved={page.url != before} method={'key' if page.url != before else 'wheel'}", flush=True)

        ctx.close()
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
