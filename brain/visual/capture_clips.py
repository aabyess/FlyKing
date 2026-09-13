"""실제 릴스에서 뇌 실험용 짧은 프레임 묶음을 저장한다(읽기 전용 — 좋아요 안 누름, 넘기기만).

  ~/flybrain/insta/.venv/bin/python brain/visual/capture_clips.py [릴스 수=6] [초=2.0] [fps=10]

산출: ~/flybrain/insta/results/clips/clip_NN.npz
  frames  uint8 (T, 256, 144, 3)  9:16 cover 맞춤 RGB
  t       float (T,)              영상 재생 시각(초)
  meta    json 문자열             url·작성자·길이·캡처 fps
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "insta"))
from common import IG, RESULTS_DIR, logged_in_user, open_fly_chrome  # noqa: E402
from reels import next_reel, read_reel  # noqa: E402

W, H = 144, 256
RAW_JS = r"""
([W, H]) => {
  const vw = innerWidth, vh = innerHeight;
  let v = null, best = 0;
  for (const el of document.querySelectorAll('video')) {
    const r = el.getBoundingClientRect();
    const a = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0)) * Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    if (a > best) { best = a; v = el; }
  }
  if (!v || v.readyState < 2 || !v.videoWidth) return null;
  const c = window.__flyRaw || (window.__flyRaw = document.createElement('canvas'));
  c.width = W; c.height = H;
  const g = c.getContext('2d', { willReadFrequently: true });
  const s = Math.max(W / v.videoWidth, H / v.videoHeight);
  g.drawImage(v, (W - v.videoWidth * s) / 2, (H - v.videoHeight * s) / 2, v.videoWidth * s, v.videoHeight * s);
  const d = g.getImageData(0, 0, W, H).data;
  let bin = '';
  for (let i = 0; i < d.length; i += 4) bin += String.fromCharCode(d[i], d[i + 1], d[i + 2]);
  return { rgb: btoa(bin), t: v.currentTime };
}
"""


def main(n=6, seconds=2.0, fps=10.0):
    import base64
    out = RESULTS_DIR / "clips"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = open_fly_chrome(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(f"{IG}/reels/")
        if not logged_in_user(ctx):
            print("NOT_LOGGED_IN", flush=True)
            return 2
        page.wait_for_selector("video", timeout=30000)
        page.wait_for_timeout(2500)
        start = len(list(out.glob("clip_*.npz")))          # 이미 있는 조각은 덮어쓰지 않고 뒤에 붙인다
        for k in range(start, start + n):
            info = read_reel(page) or {}
            frames, ts = [], []
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                t0 = time.monotonic()
                r = page.evaluate(RAW_JS, [W, H])
                if r:
                    frames.append(np.frombuffer(base64.b64decode(r["rgb"]), np.uint8).reshape(H, W, 3))
                    ts.append(r["t"])
                time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - t0)))
            meta = {"url": info.get("url"), "author": info.get("author"), "duration": info.get("duration"),
                    "fps_target": fps, "n": len(frames), "captured": time.strftime("%Y-%m-%d %H:%M:%S")}
            path = out / f"clip_{k:02d}.npz"
            np.savez_compressed(path, frames=np.stack(frames) if frames else np.zeros((0, H, W, 3), np.uint8),
                                t=np.array(ts), meta=json.dumps(meta, ensure_ascii=False))
            print("CLIP", k, json.dumps(meta, ensure_ascii=False), flush=True)
            next_reel(page)
        ctx.close()
    print("CLIPS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(int(a[0]) if a else 6, float(a[1]) if len(a) > 1 else 2.0, float(a[2]) if len(a) > 2 else 10.0))
