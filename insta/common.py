"""초파리 전용 인스타그램 브라우저 창 공통 설정.

- 로그인 정보는 코드에 넣지 않는다. 사용자가 창에서 직접 로그인하고,
  로그인 상태(쿠키)는 git 밖 프로필 폴더 ~/flybrain/insta/profile 에만 남는다.
- 이 맥에는 Google Chrome이 없어서 Playwright 전용 Chromium을 쓴다(CHANNEL=None).
  크롬을 설치하면 CHANNEL을 "chrome"으로 바꾸는 편이 봇 의심이 적다.
"""
from pathlib import Path

PROFILE_DIR = Path.home() / "flybrain" / "insta" / "profile"
RESULTS_DIR = Path.home() / "flybrain" / "insta" / "results"
IG = "https://www.instagram.com"
CHANNEL = None


def open_fly_chrome(p, headless=False):
    """초파리 계정 프로필로 브라우저를 연다. Playwright sync API의 p를 받는다."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return p.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        channel=CHANNEL,
        headless=headless,
        viewport=None,                      # 창 크기 그대로
        locale="ko-KR",
        ignore_default_args=["--enable-automation"],
        args=["--window-size=520,940"],     # 세로 릴스 비율
    )


def logged_in_user(context):
    """로그인돼 있으면 인스타 사용자 숫자 ID, 아니면 None."""
    cookies = {c["name"]: c["value"] for c in context.cookies(IG)}
    if cookies.get("sessionid"):
        return cookies.get("ds_user_id", "?")
    return None
