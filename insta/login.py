"""초파리 계정 로그인 창을 띄우고, 사용자가 직접 로그인할 때까지 기다린다.

실행: ~/flybrain/insta/.venv/bin/python insta/login.py
출력(한 줄씩): ALREADY_LOGGED_IN / WAITING_FOR_LOGIN / LOGGED_IN / CLOSED_BY_USER / TIMEOUT
"""
import sys
import time

from playwright.sync_api import sync_playwright

from common import IG, logged_in_user, open_fly_chrome

TIMEOUT_S = 20 * 60


def say(msg):
    print(msg, flush=True)


def main():
    with sync_playwright() as p:
        ctx = open_fly_chrome(p)
        closed = {"v": False}
        ctx.on("close", lambda _: closed.update(v=True))
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        page.goto(IG if logged_in_user(ctx) else f"{IG}/accounts/login/")
        uid = logged_in_user(ctx)
        if uid:
            say(f"ALREADY_LOGGED_IN user_id={uid}")
            ctx.close()
            return 0

        say("WAITING_FOR_LOGIN 크롬 창에서 초파리 계정으로 직접 로그인해 주세요")
        t0 = time.time()
        while time.time() - t0 < TIMEOUT_S:
            if closed["v"]:
                say("CLOSED_BY_USER 로그인 전에 창이 닫혔어요")
                return 2
            try:
                uid = logged_in_user(ctx)
            except Exception:
                uid = None
            if uid:
                say(f"LOGGED_IN user_id={uid}")
                page.wait_for_timeout(4000)   # 로그인 직후 쿠키 저장 시간
                ctx.close()
                return 0
            time.sleep(2)
        say("TIMEOUT 20분 안에 로그인이 끝나지 않았어요")
        ctx.close()
        return 3


if __name__ == "__main__":
    sys.exit(main())
