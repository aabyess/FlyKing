"""초파리의 손: 인스타 릴스 읽기·넘기기·좋아요.

2026-09-13 probe_reels.py 탐색 결과(한국어 웹, 1280×720 창):
- /reels/ 화면 가운데 video, 버튼(좋아요·댓글·공유·저장·더보기)은 영상 **오른쪽 바깥** 세로줄,
  작성자·캡션은 영상 **왼쪽 아래 바깥**. DOM 조상 관계로는 안 묶여서 화면 좌표로 찾는다.
- 다음 릴스: 방향키 ↓ 로 넘어간다(3/3 성공).
- 인스타 클래스 이름은 난독화돼 수시로 바뀌므로 aria-label·좌표만 쓴다.

직접 실행하면 읽기 확인만 한다(좋아요 안 누름):
  ~/flybrain/insta/.venv/bin/python insta/reels.py [릴스 개수=3]
"""
import json
import sys
import time

from common import IG, RESULTS_DIR

LIKE_LABELS = ("좋아요", "Like")
UNLIKE_LABELS = ("좋아요 취소", "Unlike")

READ_JS = r"""
() => {
  const vw = innerWidth, vh = innerHeight;
  const vis = r => Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0)) * Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
  let video = null, best = 0;
  for (const v of document.querySelectorAll('video')) { const a = vis(v.getBoundingClientRect()); if (a > best) { best = a; video = v; } }
  if (!video) return null;
  document.querySelectorAll('[data-fly]').forEach(e => e.removeAttribute('data-fly'));
  video.setAttribute('data-fly', 'video');
  const R = video.getBoundingClientRect();
  const center = el => { const r = el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2, r]; };

  // 오른쪽 버튼 줄: 영상 오른쪽 끝에서 140px 안, 영상 세로 범위 안
  const side = [...document.querySelectorAll('svg[aria-label]')].map(s => {
    const [x, y, r] = center(s); return { s, x, y, r, label: s.getAttribute('aria-label') };
  }).filter(o => o.x > R.right && o.x < R.right + 140 && o.y > R.top && o.y < R.bottom && o.r.width > 0)
    .sort((a, b) => a.y - b.y);
  const heart = side.find(o => ["좋아요", "Like", "좋아요 취소", "Unlike"].includes(o.label));
  let likeCount = null;
  if (heart) {
    const btn = heart.s.closest('[role="button"],button') || heart.s.parentElement;
    btn.setAttribute('data-fly', 'like');
    // 하트 바로 아래 숫자
    const nums = [...document.querySelectorAll('span')].map(e => { const [x, y] = center(e); return { e, x, y }; })
      .filter(o => Math.abs(o.x - heart.x) < 30 && o.y > heart.y && o.y < heart.y + 40 && o.e.children.length === 0 && o.e.innerText.trim());
    if (nums.length) likeCount = nums[0].e.innerText.trim();
  }

  // 왼쪽 아래: 작성자 링크·캡션
  const left = el => { const [x, y] = center(el); return x < R.left && y > R.bottom - 260 && y < R.bottom + 10; };
  const author = [...document.querySelectorAll('a[href^="/"]')].filter(left)
    .map(a => a.innerText.trim()).find(t => t && !t.includes('\n')) || null;
  const caption = [...document.querySelectorAll('span[dir="auto"], div[dir="auto"]')].filter(left)
    .map(e => e.innerText.trim()).filter(t => t && t !== author).sort((a, b) => b.length - a.length)[0] || null;

  return {
    url: location.href, author, caption,
    duration: video.duration, currentTime: video.currentTime, paused: video.paused, muted: video.muted,
    playbackRate: video.playbackRate, size: [video.videoWidth, video.videoHeight],
    liked: heart ? ["좋아요 취소", "Unlike"].includes(heart.label) : null,
    likeCount, sideButtons: side.map(o => o.label),
  };
}
"""


def read_reel(page):
    """지금 화면 한가운데 릴스 정보. 영상이 없으면 None."""
    return page.evaluate(READ_JS)


def capture_frame(page, path):
    """초파리 눈에 넣을 현재 영상 프레임을 PNG로 저장."""
    page.locator('video[data-fly="video"]').first.screenshot(path=str(path))


def set_playback_rate(page, rate):
    """뇌가 실시간보다 느리므로 릴스를 늦춰 맞춘다(뇌 1초 ≈ 실제 5초 → 0.2)."""
    page.evaluate('r => { const v = document.querySelector(\'video[data-fly="video"]\'); if (v) v.playbackRate = r; }', rate)


def like(page):
    """좋아요를 누른다. 이미 눌려 있으면 아무것도 안 한다. 누르면 True."""
    info = read_reel(page)
    if not info or info["liked"] is None:
        raise RuntimeError("좋아요 버튼을 못 찾음")
    if info["liked"]:
        return False
    page.locator('[data-fly="like"]').first.click()
    page.wait_for_timeout(800)
    after = read_reel(page)
    return bool(after and after["liked"])


def next_reel(page):
    """다음 릴스로 넘긴다. 주소가 바뀌면 True."""
    before = page.url
    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(2000)
    return page.url != before


def _check(n=3):
    from playwright.sync_api import sync_playwright
    from common import logged_in_user, open_fly_chrome

    out = RESULTS_DIR / "check"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = open_fly_chrome(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(f"{IG}/reels/")
        if not logged_in_user(ctx):
            print("NOT_LOGGED_IN", flush=True)
            ctx.close()
            return 2
        page.wait_for_selector("video", timeout=30000)
        page.wait_for_timeout(3000)
        for i in range(n):
            info = read_reel(page)
            set_playback_rate(page, 0.2)
            page.wait_for_timeout(300)
            info["rateAfterSet"] = read_reel(page)["playbackRate"]
            capture_frame(page, out / f"frame_{i}.png")
            info["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (out / f"reel_{i}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
            print("REEL " + json.dumps({k: info[k] for k in ("url", "author", "duration", "liked", "likeCount", "rateAfterSet", "sideButtons")}, ensure_ascii=False), flush=True)
            print(f"NEXT moved={next_reel(page)}", flush=True)
        ctx.close()
    print("CHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_check(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
