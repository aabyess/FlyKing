"""공장 게임 뇌 판단 서버 — 유니티 ⇄ 초파리 뇌. 뇌 모델 환경에서 실행한다.

  ~/flybrain/shiu-brain-model/.venv/bin/python factory/server/brain_server.py [--workers 2] [--port 8790]

구조
- 뇌 프로세스(FactoryBrain) N개. 하나당 메모리 약 4GB, 0.5초 판단 ≈ 실제 1.1초(2026-09-13 측정).
  초파리 4마리 × 2.5초 주기 = 초당 1.6판단 → 기본 2개(2개로 초당 약 1.7판단 실측).
- 판단 대기열(스케줄러): 쉬는 뇌 프로세스에만 작업을 하나씩 넘긴다. 게임 판단이 적성 검사보다 먼저다
  (적성 검사 24판단이 먼저 쌓여 게임 판단이 18초 기다리던 문제, 2026-09-13).
- 유니티와는 TCP 127.0.0.1:8790, 한 줄에 JSON 하나(UTF-8, \n).
- 판단은 매번 진짜 뇌 계산이다. 무작위는 자극 장면의 작은 위치 흔들림(seed)과 포아송 입력뿐, 판정 자체에 주사위는 없다.

유니티 → 서버
  {"type":"decide","id":"...","fly_seed":101,"station":"sorter","params":{"box":"bright|dark|none","seed":7},
   "lever_threshold":21.3,"lever_polarity":1}
  {"type":"decide","id":"...","fly_seed":101,"station":"guard","params":{"event":"intruder|clouds|none","seed":7}}
  {"type":"decide","id":"...","fly_seed":101,"station":"sugar","params":{"sugar_hz":150,"seed":7}}
  {"type":"aptitude","id":"...","fly_seed":101}          적성 검사(분류 8 + 경비 3 + 설탕 1 = 12판단)
  {"type":"status"}
서버 → 유니티
  {"type":"hello"|"status","workers_ready":2,"workers":2,"queue":0,"rules":{...}}
  {"type":"decision","id":...,"fly_seed":...,"station":...,"verdict":{...},"values":{...},"reason":"...","wall_s":..,"queue_wait_s":..}
  {"type":"aptitude","id":...,"fly_seed":...,"sorter":{...},"guard":{...},"sugar":{...}}
기록: ~/flybrain/factory/results/decisions.jsonl (판단마다 입력 특징·뉴런 발화·판정·결과)
"""
import argparse
import asyncio
import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRAIN_DIR = HERE.parent / "brain"
LOG_DIR = Path.home() / "flybrain" / "factory" / "results"
FPS = 10.0
WINDOW_S = 0.5

# 규칙 — 숫자는 factory/brain/reflex_probe.json(σ 0.1354 기본 뇌) 실측에서 왔다.
GF_THRESHOLD_HZ = 60.0            # von Reyn 2014 거대섬유 도주; 문턱은 embodied brain_body_bridge 0.3×200Hz
MN9_REF_HZ = 71.0                 # 당 감각뉴런 200Hz 때 MN9 평균(reflex_probe sugar)
SORTER_DEFAULT_THRESHOLD = 21.3   # 적성 검사 전 기본 레버 문턱 = 밝은 상자 22.5·어두운 상자 20.1Hz 평균의 가운데
RULES = {
    "sorter": {"neuron": "다가가기 oDN1·P9 평균 발화",
               "rule": "≥ 이 초파리 레버 문턱이면 레버를 밀어 상자를 A통(밝은 상자)으로. 어두운 상자에 더 다가가는 개체는 공장이 레버를 반대로 연결",
               "source": "Bidaye et al. 2020 Neuron 108:469 — 전진 명령 뉴런"},
    "guard": {"neuron": "거대섬유 DNp01 50ms 창 최고 발화", "rule": f"≥ {GF_THRESHOLD_HZ}Hz면 경보", "threshold": GF_THRESHOLD_HZ,
              "source": "von Reyn et al. 2014 Nat Neurosci 17:962 — 루밍 도약 도주"},
    "sugar": {"neuron": "MN9 주둥이 운동뉴런 평균 발화", "rule": f"÷ {MN9_REF_HZ}Hz(당 200Hz 때) = 운반 속도 배수(최대 1.5)",
              "ref": MN9_REF_HZ, "source": "Shiu et al. 2024 Nature — 당 → 섭식 운동뉴런"},
}


def log(*a):
    print(*a, flush=True)


# ---------------- 뇌 프로세스 ----------------
def worker_main(jobs, results, wid):
    sys.path.insert(0, str(BRAIN_DIR))
    import stimuli
    from fbrain import FactoryBrain

    brain = FactoryBrain(fps=FPS)
    results.put({"kind": "ready", "worker": wid, "build_s": brain.build_s, "sigma": brain.sigma})
    while True:
        job = jobs.get()
        if job is None:
            break
        try:
            st, p = job["station"], job.get("params", {})
            seed = int(p.get("seed", 0))
            sugar_hz = 0.0
            if st == "sorter":
                frames = stimuli.sorter(p.get("box", "none"), WINDOW_S, FPS, seed)
            elif st == "guard":
                frames = stimuli.guard(p.get("event", "none"), WINDOW_S, FPS, seed)
            elif st == "sugar":
                frames = stimuli.sugar(WINDOW_S, FPS, seed)
                sugar_hz = float(p.get("sugar_hz", 0.0))
            else:
                raise ValueError(f"작업대 모름: {st}")
            t_start = time.time()
            res = brain.decide(frames, sugar_hz=sugar_hz, fly_seed=job["fly_seed"], seed=seed)
            results.put({"kind": "result", "worker": wid, "job": job, "res": res, "started": t_start})
        except Exception as e:  # noqa: BLE001
            results.put({"kind": "error", "worker": wid, "job": job, "error": repr(e)})


# ---------------- 판정 ----------------
def judge(job, res):
    st, p, v = job["station"], job.get("params", {}), res["values"]
    if st == "sorter":
        thr = float(job.get("lever_threshold") or SORTER_DEFAULT_THRESHOLD)
        # 레버 방향: 적성 검사에서 어두운 상자에 더 다가가는 개체면 공장이 레버를 반대로 연결(-1). 초파리는 그대로다.
        polarity = -1 if float(job.get("lever_polarity") or 1) < 0 else 1
        box = p.get("box", "none")
        above = v["approach_hz"] >= thr
        push = above if polarity > 0 else not above
        correct = (box == "bright" and push) or (box != "bright" and not push)
        outcome = {"bright": "A통 맞게 분류" if push else "밝은 상자를 놓침(B통)",
                   "dark": "어두운 상자를 A통에 잘못 넣음" if push else "B통 맞게 통과",
                   "none": "빈 칸인데 레버를 밀어 헛동작" if push else "빈 칸 통과"}[box]
        wiring = "" if polarity > 0 else "(반대 연결) "
        reason = f"다가가기 {v['approach_hz']}Hz {'≥' if above else '<'} 레버 문턱 {thr}Hz {wiring}→ {'레버 밀기' if push else '안 밂'}"
        return {"action": "push" if push else "pass", "correct": bool(correct), "threshold": thr, "polarity": polarity,
                "outcome": outcome}, reason
    if st == "guard":
        ev = p.get("event", "none")
        alarm = v["GF_peak50ms_hz"] >= GF_THRESHOLD_HZ
        outcome = ("침입자 막음" if alarm else "침입자 놓침") if ev == "intruder" else ("헛경보" if alarm else "조용히 지킴")
        reason = f"거대섬유 최고 {v['GF_peak50ms_hz']}Hz {'≥' if alarm else '<'} {GF_THRESHOLD_HZ}Hz → {'경보' if alarm else '경보 없음'}"
        return {"action": "alarm" if alarm else "quiet", "correct": bool(alarm == (ev == "intruder")), "threshold": GF_THRESHOLD_HZ,
                "outcome": outcome}, reason
    speed = round(min(1.5, v["MN9_mean_hz"] / MN9_REF_HZ), 3)
    reason = f"당 {p.get('sugar_hz', 0)}Hz 자극 → MN9 {v['MN9_mean_hz']}Hz ÷ {MN9_REF_HZ}Hz → 운반 속도 ×{speed}"
    return {"action": "carry", "speed": speed, "correct": speed > 0, "threshold": MN9_REF_HZ, "outcome": f"속도 ×{speed}"}, reason


# ---------------- 서버 ----------------
class Server:
    def __init__(self, n_workers):
        self.ctx = mp.get_context("spawn")
        self.results = self.ctx.Queue()
        self.n_workers = n_workers
        self.worker_q = [self.ctx.Queue() for _ in range(n_workers)]
        self.worker_ready = [False] * n_workers
        self.busy = [False] * n_workers
        self.high, self.low = deque(), deque()     # 게임 판단 / 적성 검사
        self.pending = {}                          # job key → writer
        self.aptitudes = {}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = open(LOG_DIR / "decisions.jsonl", "a")

    @property
    def ready(self):
        return sum(self.worker_ready)

    @property
    def queued(self):
        return len(self.high) + len(self.low) + sum(self.busy)

    def start_workers(self):
        # Cython 컴파일 캐시가 동시에 쓰이지 않게 하나씩 띄운다(앞 프로세스가 ready 보고한 뒤 다음).
        def launcher():
            for wid in range(self.n_workers):
                self.ctx.Process(target=worker_main, args=(self.worker_q[wid], self.results, wid), daemon=True).start()
                while not self.worker_ready[wid]:
                    time.sleep(0.2)
        threading.Thread(target=launcher, daemon=True).start()

    def status(self):
        return {"type": "status", "workers_ready": self.ready, "workers": self.n_workers, "queue": self.queued,
                "waiting_game": len(self.high), "waiting_aptitude": len(self.low), "window_s": WINDOW_S, "rules": RULES,
                "sorter_default_threshold": SORTER_DEFAULT_THRESHOLD}

    async def send(self, writer, msg):
        if writer is None:
            return
        try:
            writer.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
            await writer.drain()
        except (ConnectionError, RuntimeError):
            pass

    def enqueue(self, writer, job):
        job["queued_at"] = time.time()
        self.pending[job["key"]] = writer
        (self.low if job.get("aptitude") else self.high).append(job)
        self.dispatch()

    def dispatch(self):
        for wid in range(self.n_workers):
            if not self.worker_ready[wid] or self.busy[wid]:
                continue
            if not self.high and not self.low:
                return
            job = self.high.popleft() if self.high else self.low.popleft()
            self.busy[wid] = True
            self.worker_q[wid].put(job)

    async def pump_results(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, self.results.get, True, 0.5)
            except queue.Empty:
                continue
            wid = msg["worker"]
            if msg["kind"] == "ready":
                self.worker_ready[wid] = True
                log("WORKER_READY", wid, "build", msg["build_s"], "sigma", msg["sigma"])
                self.dispatch()
                continue
            self.busy[wid] = False
            self.dispatch()
            job = msg["job"]
            writer = self.pending.pop(job["key"], None)
            if msg["kind"] == "error":
                log("JOB_ERROR", msg["error"])
                await self.send(writer, {"type": "error", "id": job.get("id"), "error": msg["error"]})
                continue
            res = msg["res"]
            verdict, reason = judge(job, res)
            record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "id": job.get("id"), "purpose": job.get("purpose", "game"),
                      "fly_seed": job["fly_seed"], "station": job["station"], "params": job.get("params"),
                      "lever_threshold": job.get("lever_threshold"), "lever_polarity": job.get("lever_polarity"),
                      "verdict": verdict, "reason": reason, "rule": RULES[job["station"]], "values": res["values"],
                      "features": res["features"], "vpn_rate_hz": res["vpn_rate_hz"], "input": res["input"],
                      "sigma": res["sigma"], "washout_spikes": res["washout_spikes"], "wall_s": res["wall_s"],
                      "queue_wait_s": round(msg["started"] - job["queued_at"], 3), "worker": wid}
            self.log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.log_file.flush()
            if job.get("aptitude"):
                await self.collect_aptitude(job, record)
            else:
                await self.send(writer, {"type": "decision", **{k: record[k] for k in (
                    "id", "fly_seed", "station", "params", "verdict", "reason", "values", "wall_s", "queue_wait_s", "worker")}})

    async def collect_aptitude(self, job, record):
        a = self.aptitudes.get(job["aptitude"])
        if a is None:
            return
        a["records"].append(record)
        if len(a["records"]) < a["total"]:
            return
        recs = a["records"]
        bright = [r["values"]["approach_hz"] for r in recs if r["station"] == "sorter" and r["params"]["box"] == "bright"]
        dark = [r["values"]["approach_hz"] for r in recs if r["station"] == "sorter" and r["params"]["box"] == "dark"]
        mb, md = sum(bright) / len(bright), sum(dark) / len(dark)
        var = (sum((x - mb) ** 2 for x in bright) + sum((x - md) ** 2 for x in dark)) / max(len(bright) + len(dark) - 2, 1)
        sd = max(var ** 0.5, 1e-6)
        thr = round((mb + md) / 2, 2)
        polarity = 1 if mb >= md else -1          # 어두운 상자에 더 다가가는 개체는 레버를 반대로 연결
        hit = (lambda x: x >= thr) if polarity > 0 else (lambda x: x < thr)
        acc = (sum(hit(x) for x in bright) + sum(not hit(x) for x in dark)) / (len(bright) + len(dark))
        gf = [r["values"]["GF_peak50ms_hz"] for r in recs if r["station"] == "guard"]
        mn9 = [r["values"]["MN9_mean_hz"] for r in recs if r["station"] == "sugar"]
        out = {"type": "aptitude", "id": a["id"], "fly_seed": a["fly_seed"],
               "sorter": {"bright_mean_hz": round(mb, 2), "dark_mean_hz": round(md, 2), "lever_threshold": thr,
                          "lever_polarity": polarity, "dprime": round(abs(mb - md) / sd, 2), "test_accuracy": round(acc, 3)},
               "guard": {"intruder_GF_hz": gf, "detect_rate": round(sum(g >= GF_THRESHOLD_HZ for g in gf) / len(gf), 3)},
               "sugar": {"MN9_at_150hz": mn9[0] if mn9 else None,
                         "speed_at_150hz": round(min(1.5, mn9[0] / MN9_REF_HZ), 3) if mn9 else None},
               "note": "적성 검사 — 초파리가 배우는 게 아니라 공장이 레버 감도·연결을 이 개체 반응에 맞춘다"}
        del self.aptitudes[job["aptitude"]]
        log("APTITUDE", json.dumps(out, ensure_ascii=False))
        await self.send(a["writer"], out)

    async def handle(self, reader, writer):
        log("CLIENT_CONNECTED")
        await self.send(writer, {**self.status(), "type": "hello"})
        n = 0
        while True:
            try:
                line = await reader.readline()
            except (ConnectionError, asyncio.LimitOverrunError):
                break
            if not line:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            n += 1
            t = msg.get("type")
            if t == "status":
                await self.send(writer, self.status())
            elif t == "decide":
                job = {k: msg.get(k) for k in ("id", "fly_seed", "station", "params", "lever_threshold", "lever_polarity")}
                job["key"] = f"{msg.get('id')}#{time.time()}#{n}"
                self.enqueue(writer, job)
            elif t == "aptitude":
                aid = f"apt#{msg.get('id')}#{time.time()}"
                plan = [("sorter", {"box": b, "seed": s}) for b in ("bright", "dark") for s in range(4)]
                plan += [("guard", {"event": "intruder", "seed": s}) for s in range(3)] + [("sugar", {"sugar_hz": 150, "seed": 0})]
                self.aptitudes[aid] = {"id": msg.get("id"), "fly_seed": msg["fly_seed"], "writer": writer, "total": len(plan), "records": []}
                for i, (st, p) in enumerate(plan):
                    self.enqueue(writer, {"id": f"{msg.get('id')}-{i}", "fly_seed": msg["fly_seed"], "station": st, "params": p,
                                          "aptitude": aid, "purpose": "aptitude", "key": f"{aid}#{i}"})
        log("CLIENT_LEFT")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    srv = Server(a.workers)
    srv.start_workers()
    server = await asyncio.start_server(srv.handle, "127.0.0.1", a.port, limit=16 * 1024 * 1024)
    log(f"FACTORY_SERVER_READY 127.0.0.1:{a.port} workers={a.workers} (뇌 프로세스는 하나씩 올라오는 중)")
    asyncio.create_task(srv.pump_results())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
