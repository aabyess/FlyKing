"""유니티 없이 뇌 판단 서버를 끝까지 시험한다(아무 파이썬).

  python3 factory/server/test_client.py [--port 8790]

순서: 뇌 프로세스 준비 기다림 → 초파리 2마리 적성 검사 동시 요청 → 작업대 판단 9개(분류·경비·설탕 섞어) 동시 요청
→ 응답 시간·대기·판정을 출력.
"""
import argparse
import json
import socket
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    s = socket.create_connection(("127.0.0.1", a.port))
    f = s.makefile("rwb")

    def send(msg):
        f.write((json.dumps(msg) + "\n").encode())
        f.flush()

    def recv():
        line = f.readline()
        return json.loads(line) if line else None

    hello = recv()
    print("HELLO", hello["workers_ready"], "/", hello["workers"], flush=True)
    while True:
        send({"type": "status"})
        st = recv()
        if st["workers_ready"] >= st["workers"]:
            break
        time.sleep(2)
    print("WORKERS_READY", st["workers_ready"], flush=True)

    t0 = time.time()
    for fly in (101, 102):
        send({"type": "aptitude", "id": f"apt{fly}", "fly_seed": fly})
    jobs = [("sorter", {"box": "bright", "seed": 11}), ("sorter", {"box": "dark", "seed": 12}), ("sorter", {"box": "none", "seed": 13}),
            ("guard", {"event": "intruder", "seed": 21}), ("guard", {"event": "clouds", "seed": 22}), ("guard", {"event": "none", "seed": 23}),
            ("sugar", {"sugar_hz": 100, "seed": 31}), ("sugar", {"sugar_hz": 150, "seed": 32}), ("sugar", {"sugar_hz": 200, "seed": 33})]
    for i, (st_name, p) in enumerate(jobs):
        send({"type": "decide", "id": f"d{i}", "fly_seed": 101 + i % 2, "station": st_name, "params": p})
    want = 2 + len(jobs)
    got = 0
    while got < want:
        m = recv()
        if m is None:
            break
        if m["type"] == "decision":
            got += 1
            print("DECISION", m["id"], m["station"], json.dumps(m["params"]), "→", m["verdict"]["outcome"], "|", m["reason"],
                  f"| 계산 {m['wall_s']}s 대기 {m['queue_wait_s']}s 뇌#{m['worker']}", flush=True)
        elif m["type"] == "aptitude":
            got += 1
            print("APTITUDE", json.dumps(m, ensure_ascii=False), flush=True)
        elif m["type"] == "error":
            got += 1
            print("ERROR", m, flush=True)
    print("TOTAL_WALL", round(time.time() - t0, 1), "s for", want, "responses (적성 12판단×2 + 판단 9 = 33판단)", flush=True)
    s.close()


if __name__ == "__main__":
    main()
