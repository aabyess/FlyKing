"""뷰어 자동 시험 — 허브가 떠 있는 상태에서 헤드리스 브라우저로 뷰어를 열고 순간들을 캡처한다.

  ~/flybrain/insta/.venv/bin/python live/hub.py --source=mock --like-prob=1.0 &
  ~/flybrain/insta/.venv/bin/python live/test_viewer.py [--live=25]

산출: ~/flybrain/insta/results/viewer/*.png
  walk / watch_wide / like_touch_side / like_touch_shoulder / swipe_drag_shoulder / live_*.png(허브 흐름대로 실제 진행 중 캡처)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path.home() / "flybrain" / "insta" / "results" / "viewer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--live", type=float, default=25.0, help="허브 흐름대로 지켜볼 초")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = b.new_page(viewport={"width": 1280, "height": 800})
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
        page.goto(a.url)
        try:
            page.wait_for_function("window.fly && window.fly.ready", timeout=120000)
        except Exception:  # noqa: BLE001
            print("NOT_READY", json.dumps(logs[-15:], ensure_ascii=False), flush=True)
            page.screenshot(path=str(OUT / "not_ready.png"))
            b.close()
            return 2
        print("FLY_INFO", json.dumps(page.evaluate("window.fly.info")), flush=True)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "walk.png"))

        def shot(name, cam, freeze=None, wait=900):
            if freeze:
                page.evaluate("([a, t]) => window.fly.freeze(a, t)", freeze)
            else:
                page.evaluate("window.fly.unfreeze(); window.fly.skipIntro()")
            page.evaluate("(c) => window.fly.setCam(c)", cam)
            page.wait_for_timeout(wait)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print("SHOT", name, flush=True)

        page.evaluate("window.fly.skipIntro()")
        page.wait_for_timeout(1500)
        shot("watch_wide", "wide")
        shot("like_touch_side", "side", ["like", 1.0])
        shot("like_touch_shoulder", "shoulder", ["like", 1.0])
        shot("swipe_drag_shoulder", "shoulder", ["swipe", 1.35])
        shot("swipe_drag_side", "side", ["swipe", 1.35])
        page.evaluate("window.fly.unfreeze()")
        page.evaluate("window.fly.setCam('shoulder')")

        t0, k = time.time(), 0
        while time.time() - t0 < a.live:
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / f"live_{k:02d}.png"))
            st = page.evaluate("({a: window.fly.state.action && window.fly.state.action.m.action, liked: window.fly.state.liked, reel: window.fly.state.reel && window.fly.state.reel.id, seen: window.fly.state.seen})")
            print("LIVE", k, json.dumps(st), flush=True)
            k += 1
        errs = [l for l in logs if l.startswith(("error", "PAGEERROR"))]
        print("CONSOLE_ERRORS", json.dumps(errs[:10], ensure_ascii=False), flush=True)
        b.close()
    print("VIEWER_TEST_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
