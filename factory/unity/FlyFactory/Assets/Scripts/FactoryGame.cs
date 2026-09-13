using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// 초파리 뇌로 움직이는 공장(방치형) — 게임 규칙·경제·화면.
///
/// 원칙: 초파리는 조종도 학습도 되지 않는다. 일은 Shiu 2024 전뇌 모델에서 재현되는 타고난 반사로만 한다.
///   불빛 분류대 = 다가가기 명령 뉴런(oDN1·P9) / 경비 초소 = 거대섬유 도주(DNp01) / 설탕 배달대 = 당 → 섭식 운동뉴런(MN9).
///   판정은 매번 뇌 서버의 진짜 계산 결과다. 여기서 쓰는 난수는 환경(어떤 상자가 오나·침입자가 언제 오나)뿐이다.
/// 플레이어가 하는 일: 초파리를 어느 작업대에 둘지, 작업대·초파리 구입, 설탕 농도, 분류대 레버 감도.
/// 경제 수치 근거: factory/brain/reflex_probe.json 실측(분류 정확도 약 58~75%, 경비 침입자 검출, 설탕 곡선).
/// </summary>
public enum StationKind { Sorter, Guard, Sugar }

public class Fly
{
    public int seed;
    public string name;
    public Station station;
    public bool pending;
    public float pendingSince, nextAt;
    public Dictionary<string, object> values;
    public string reason = "아직 판단 전";
    public string outcome = "";
    public bool aptDone;
    public string aptText = "적성 검사 중(뇌 계산 12번)";
    public double leverThreshold = 21.3, dprime, testAcc, detect, speed150;
    public int leverPolarity = 1;
    public GameObject go;
    public Animation anim;
    public string idleClip, walkClip;
    public float walkUntil;
    public int decisions;
}

public class Station
{
    public StationKind kind;
    public int number;
    public Fly fly;
    public GameObject root;
    public Transform flySpot, lever, boxBright, boxDark, boxStart, boxEnd, binA, binB, lamp, gate, intruder, cart, cartStart, cartEnd;
    public int correct, wrong, blocked, missed, falseAlarm, deliveries, noReaction;
    public double earned;
    // 분류대
    public int boxPhase;           // 0 대기 · 1 레버 앞으로 · 2 뇌 판단 기다림 · 3 통으로
    public float boxT;
    public bool decided, push;
    public string boxKind;
    public Transform boxObj;
    public Vector3 boxFrom, boxTo;
    public float leverUntil;
    public Quaternion leverRest;
    // 경비
    public float lampUntil, gateUntil;
    public Vector3 gateOpen, intruderBase = Vector3.one;
    // 설탕
    public int cartPhase;          // 0 대기 · 1 배달 중 · 2 돌아옴
    public float cartT, cartDur;

    public string Name => kind == StationKind.Sorter ? $"불빛 분류대 {number}" : kind == StationKind.Guard ? $"경비 초소 {number}" : $"설탕 배달대 {number}";
}

[RequireComponent(typeof(BrainClient))]
public class FactoryGame : MonoBehaviour
{
    const double SortRight = 4, SortWrong = -2, IntruderLoss = -30, FalseAlarmLoss = -3, DeliveryPay = 10, SugarCostPerHz = 0.02;
    const float Cycle = 2.5f, Spacing = 300f, IntruderWindow = 4f, IntruderMean = 20f, GfThreshold = 60f, Mn9Ref = 71f;

    public double money = 200;
    public float sugarHz = 150f;

    readonly List<Fly> flies = new List<Fly>();
    readonly List<Station> stations = new List<Station>();
    readonly Dictionary<string, (Fly fly, Station st)> waiting = new Dictionary<string, (Fly, Station)>();
    readonly Queue<(float t, double v)> income = new Queue<(float, double)>();
    readonly System.Random env = new System.Random(20260913);
    readonly string[] names = { "누리", "보리", "호박", "초코", "깨비", "망고", "두부", "율무", "콩이", "모카", "단지", "쑥이" };

    BrainClient brain;
    int nextSeed = 101, requestN, fliesBought, stationsBought;
    bool intruderActive, intruderBlocked;
    float intruderStart, nextIntruderAt;
    string eventText = "공장 가동 준비 중";
    Camera cam;
    Font font;
    GUIStyle title, body, small;
    Vector2 scroll;
    string shotPath;
    float shotAt = -1f;
    bool shotTaken;

    // ---------------- 시작 ----------------
    void Start()
    {
        Application.runInBackground = true;
        brain = GetComponent<BrainClient>();
        brain.OnMessage += OnBrain;
        ParseArgs();
        BuildWorld();
        AddStation(StationKind.Sorter);
        AddStation(StationKind.Sorter);
        AddStation(StationKind.Guard);
        AddStation(StationKind.Sugar);
        for (int i = 0; i < 4; i++) AddFly(stations[i]);
        nextIntruderAt = Time.time + Exp(IntruderMean);
    }

    void ParseArgs()
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "-shot") shotPath = args[i + 1];
            if (args[i] == "-shotAfter" && float.TryParse(args[i + 1], out float s)) shotAt = s;
        }
    }

    float Exp(float mean) => (float)(-mean * Math.Log(1.0 - env.NextDouble()));

    void BuildWorld()
    {
        cam = Camera.main;
        if (cam == null)
        {
            var c = new GameObject("카메라");
            cam = c.AddComponent<Camera>();
            c.tag = "MainCamera";
        }
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.12f, 0.11f, 0.10f);
        cam.fieldOfView = 38f;
        cam.farClipPlane = 6000f;
        cam.nearClipPlane = 1f;

        var sun = new GameObject("햇빛").AddComponent<Light>();
        sun.type = LightType.Directional;
        sun.intensity = 1.1f;
        sun.shadows = LightShadows.Soft;
        sun.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        RenderSettings.ambientLight = new Color(0.42f, 0.40f, 0.38f);

        var under = GameObject.CreatePrimitive(PrimitiveType.Plane);
        under.name = "바닥_바탕";
        under.transform.localScale = new Vector3(600f, 1f, 600f);
        under.transform.position = new Vector3(0f, -2.5f, 0f);
        under.GetComponent<Renderer>().material.color = new Color(0.22f, 0.17f, 0.12f);
        worldRoot = new GameObject("공장 바닥·벽").transform;
        RebuildFloor();
    }

    Transform worldRoot;

    void RebuildFloor()
    {
        // Blender 바닥 타일(100×100, 이어도 무늬 안 끊김)·벽 한 칸(폭 100)으로 작업대 줄 크기에 맞춰 깐다.
        foreach (Transform c in worldRoot) Destroy(c.gameObject);
        var tile = Resources.Load<GameObject>("Models/floor_tile");
        var wall = Resources.Load<GameObject>("Models/wall");
        int rows = Mathf.Max(4, stations.Count) * 3 + 2;
        float z0 = -2 * 100f;
        for (int ix = -3; ix <= 2; ix++)
            for (int iz = 0; iz < rows; iz++)
                if (tile) Instantiate(tile, new Vector3(ix * 100f + 50f, 0f, z0 + iz * 100f + 50f), Quaternion.identity, worldRoot);
        for (int iz = 0; iz < rows; iz++)
            if (wall) Instantiate(wall, new Vector3(250f, 0f, z0 + iz * 100f + 50f), Quaternion.Euler(0f, 90f, 0f), worldRoot);
    }

    // ---------------- 작업대 ----------------
    Station AddStation(StationKind kind)
    {
        var st = new Station { kind = kind, number = stations.Count(s => s.kind == kind) + 1 };
        string asset = kind == StationKind.Sorter ? "station_sorter" : kind == StationKind.Guard ? "station_guard" : "station_sugar";
        var prefab = Resources.Load<GameObject>("Models/" + asset);
        st.root = prefab != null ? Instantiate(prefab) : Fallback(kind);
        st.root.name = st.Name;
        stations.Add(st);
        st.root.transform.position = new Vector3(0f, 0f, (stations.Count - 1) * Spacing);
        Transform T(string n) => FindDeep(st.root.transform, n);
        st.flySpot = T("초파리_자리");
        st.lever = T("레버");
        st.boxBright = T("상자_밝음");
        st.boxDark = T("상자_어두움");
        st.boxStart = T("상자_시작");
        st.boxEnd = T("상자_끝");
        st.binA = T("통_A_입구");
        st.binB = T("통_B_입구");
        st.lamp = T("경보등");
        st.gate = T("차단문");
        st.intruder = T("침입자");
        st.cart = T("수레");
        st.cartStart = T("수레_시작");
        st.cartEnd = T("수레_끝");
        if (st.lever) st.leverRest = st.lever.localRotation;
        if (st.gate) st.gateOpen = st.gate.localPosition;
        if (st.boxBright) st.boxBright.gameObject.SetActive(false);
        if (st.boxDark) st.boxDark.gameObject.SetActive(false);
        if (st.intruder)
        {
            st.intruderBase = st.intruder.localScale;
            st.intruder.gameObject.SetActive(false);
        }
        FrameCamera();
        if (worldRoot != null) RebuildFloor();
        return st;
    }

    static Transform FindDeep(Transform t, string name)
    {
        if (t.name == name) return t;
        foreach (Transform c in t)
        {
            var r = FindDeep(c, name);
            if (r) return r;
        }
        return null;
    }

    GameObject Fallback(StationKind kind)
    {
        // 설비 FBX가 아직 없을 때 쓰는 기본 도형(Blender 모델이 들어오면 자동으로 대체된다). 소켓 이름은 FBX와 같다.
        var root = new GameObject();
        GameObject Box(string n, Vector3 pos, Vector3 size, Color c, Transform parent = null)
        {
            var g = GameObject.CreatePrimitive(PrimitiveType.Cube);
            g.name = n;
            g.transform.SetParent(parent ? parent : root.transform, false);
            g.transform.localPosition = pos;
            g.transform.localScale = size;
            g.GetComponent<Renderer>().material.color = c;
            return g;
        }
        void Socket(string n, Vector3 pos)
        {
            var s = new GameObject(n);
            s.transform.SetParent(root.transform, false);
            s.transform.localPosition = pos;
        }
        var metal = new Color(0.55f, 0.56f, 0.58f);
        var wood = new Color(0.55f, 0.40f, 0.26f);
        if (kind == StationKind.Sorter)
        {
            Box("벨트", new Vector3(0, 7.5f, 0), new Vector3(160, 15, 40), new Color(0.25f, 0.25f, 0.27f));
            var lever = new GameObject("레버");
            lever.transform.SetParent(root.transform, false);
            lever.transform.localPosition = new Vector3(-50, 4.5f, -28);
            Box("레버_막대", new Vector3(0, 6, 0), new Vector3(2, 12, 2), metal, lever.transform);
            Box("통_A", new Vector3(-50, 10, -60), new Vector3(30, 20, 30), wood);
            Box("통_B", new Vector3(100, 10, 0), new Vector3(30, 20, 44), wood);
            Box("상자_밝음", new Vector3(-70, 24, 0), new Vector3(18, 18, 18), new Color(0.92f, 0.92f, 0.92f));
            Box("상자_어두움", new Vector3(-70, 24, 0), new Vector3(18, 18, 18), new Color(0.08f, 0.08f, 0.08f));
            Socket("초파리_자리", new Vector3(-108, 1.5f, 0));
            Socket("상자_시작", new Vector3(-70, 24, 0));
            Socket("상자_끝", new Vector3(80, 24, 0));
            Socket("통_A_입구", new Vector3(-50, 24, -60));
            Socket("통_B_입구", new Vector3(100, 24, 0));
        }
        else if (kind == StationKind.Guard)
        {
            Box("받침", new Vector3(0, 3, 0), new Vector3(60, 6, 60), wood);
            Box("기둥", new Vector3(25, 35, 25), new Vector3(4, 70, 4), metal);
            Box("경보등", new Vector3(25, 74, 25), new Vector3(10, 8, 10), new Color(0.5f, 0.1f, 0.1f));
            Box("차단문", new Vector3(45, 30, 0), new Vector3(4, 50, 60), metal);
            var disk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disk.name = "침입자";
            disk.transform.SetParent(root.transform, false);
            disk.transform.localPosition = new Vector3(0, 110, 0);
            disk.transform.localScale = new Vector3(30, 1, 30);
            disk.GetComponent<Renderer>().material.color = new Color(0.05f, 0.05f, 0.06f);
            Socket("초파리_자리", new Vector3(0, 6, 0));
        }
        else
        {
            var dish = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            dish.name = "설탕";
            dish.transform.SetParent(root.transform, false);
            dish.transform.localPosition = new Vector3(-40, 1.5f, 0);
            dish.transform.localScale = new Vector3(20, 1.5f, 20);
            dish.GetComponent<Renderer>().material.color = new Color(0.95f, 0.95f, 0.92f);
            Box("레일", new Vector3(20, 1, 0), new Vector3(120, 2, 10), metal);
            Box("선반", new Vector3(90, 12, 0), new Vector3(20, 24, 40), wood);
            Box("수레", new Vector3(-30, 6, 0), new Vector3(20, 10, 16), new Color(0.35f, 0.5f, 0.7f));
            Socket("초파리_자리", new Vector3(-70, 0, 0));
            Socket("수레_시작", new Vector3(-30, 6, 0));
            Socket("수레_끝", new Vector3(70, 6, 0));
        }
        return root;
    }

    void FrameCamera()
    {
        float center = (stations.Count - 1) * Spacing / 2f;
        float span = Mathf.Max(1, stations.Count) * Spacing;
        cam.transform.position = new Vector3(-0.62f * span - 260f, 0.42f * span + 220f, center - 60f);
        cam.transform.LookAt(new Vector3(20f, 10f, center));
    }

    // ---------------- 초파리 ----------------
    Fly AddFly(Station st)
    {
        var f = new Fly { seed = nextSeed++, name = names[flies.Count % names.Length] };
        flies.Add(f);
        var prefab = Resources.Load<GameObject>("Models/초파리");
        if (prefab != null)
        {
            f.go = Instantiate(prefab);
            f.go.transform.localScale = Vector3.one * 10f;   // body FBX는 1 = 1mm, 공장 장면은 초파리 ×10
        }
        else
        {
            f.go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            f.go.transform.localScale = new Vector3(10, 6, 10);
        }
        f.go.name = $"초파리 {f.name}";
        f.anim = f.go.GetComponentInChildren<Animation>();
        if (f.anim != null)
        {
            foreach (AnimationState s in f.anim)
            {
                if (s.name.Contains("Idle_Groom")) f.idleClip = s.name;
                if (s.name.Contains("Walk_Tripod")) f.walkClip = s.name;
                s.wrapMode = WrapMode.Loop;
            }
            if (f.idleClip != null) f.anim.Play(f.idleClip);
        }
        Assign(f, st);
        brain_SendAptitude(f);
        return f;
    }

    void brain_SendAptitude(Fly f)
    {
        f.aptDone = false;
        f.aptText = "적성 검사 중(뇌 계산 12번)";
        if (brain != null && brain.Connected)
            brain.Send(new Dictionary<string, object> { ["type"] = "aptitude", ["id"] = $"apt{f.seed}", ["fly_seed"] = f.seed });
        else
            f.aptText = "뇌 서버 연결되면 적성 검사";
    }

    void Assign(Fly f, Station st)
    {
        if (f.station != null) f.station.fly = null;
        if (st != null && st.fly != null) st.fly.station = null;
        f.station = st;
        if (st != null) st.fly = f;
        PlaceFlies();
    }

    void PlaceFlies()
    {
        int idle = 0;
        foreach (var f in flies)
        {
            if (f.station != null)
            {
                var st = f.station;
                Vector3 spot = st.flySpot ? st.flySpot.position : st.root.transform.position + new Vector3(-80, 0, 0);
                Face(f, spot, st.root.transform.position + new Vector3(0, 0, 0));
            }
            else
            {
                Vector3 spot = new Vector3(-260f, 0f, -120f - idle * 70f);
                Face(f, spot, spot + new Vector3(60, 0, 0));
                idle++;
            }
        }
    }

    static void Face(Fly f, Vector3 pos, Vector3 target)
    {
        // 모델의 머리 방향을 뼈(Thorax → Head)에서 읽어, 그 방향이 target을 보게 돌린다(FBX 축 변환에 기대지 않음).
        f.go.transform.rotation = Quaternion.identity;
        f.go.transform.position = pos;
        var head = FindDeep(f.go.transform, "Head");
        var thorax = FindDeep(f.go.transform, "Thorax");
        Vector3 fwd = head && thorax ? head.position - thorax.position : Vector3.right;
        fwd.y = 0;
        Vector3 want = target - pos;
        want.y = 0;
        if (fwd.sqrMagnitude > 1e-6f && want.sqrMagnitude > 1e-6f)
            f.go.transform.rotation = Quaternion.FromToRotation(fwd.normalized, want.normalized);
    }

    // ---------------- 뇌 요청 ----------------
    void Request(Station st, string station, Dictionary<string, object> p)
    {
        var f = st.fly;
        string id = $"r{++requestN}";
        p["seed"] = (double)(requestN * 7 + f.seed);
        var msg = new Dictionary<string, object> { ["type"] = "decide", ["id"] = id, ["fly_seed"] = (double)f.seed, ["station"] = station, ["params"] = p };
        if (st.kind == StationKind.Sorter)
        {
            msg["lever_threshold"] = f.leverThreshold;
            msg["lever_polarity"] = (double)f.leverPolarity;
        }
        waiting[id] = (f, st);
        f.pending = true;
        f.pendingSince = Time.time;
        brain.Send(msg);
    }

    void OnBrain(Dictionary<string, object> m)
    {
        string type = MiniJson.Text(m, "type");
        if (type == "hello")
        {
            foreach (var f in flies.Where(x => !x.aptDone)) brain_SendAptitude(f);
            return;
        }
        if (type == "aptitude")
        {
            var f = flies.FirstOrDefault(x => x.seed == (int)MiniJson.Num(m, "fly_seed"));
            if (f == null) return;
            var so = MiniJson.Obj(m, "sorter");
            var gu = MiniJson.Obj(m, "guard");
            var su = MiniJson.Obj(m, "sugar");
            f.leverThreshold = MiniJson.Num(so, "lever_threshold", 21.3);
            f.leverPolarity = MiniJson.Num(so, "lever_polarity", 1) < 0 ? -1 : 1;
            f.dprime = MiniJson.Num(so, "dprime");
            f.testAcc = MiniJson.Num(so, "test_accuracy");
            f.detect = MiniJson.Num(gu, "detect_rate");
            f.speed150 = MiniJson.Num(su, "speed_at_150hz");
            f.aptDone = true;
            string wiring = f.leverPolarity < 0 ? " · 어두운 상자에 더 다가가서 레버 반대 연결" : "";
            f.aptText = $"적성: 밝기 구별 d′ {f.dprime:0.00} (검사 정확도 {f.testAcc * 100:0}%){wiring} · 침입자 검출 {f.detect * 100:0}% · 설탕 150Hz 속도 ×{f.speed150:0.00}";
            return;
        }
        if (type != "decision" && type != "error") return;
        string id = MiniJson.Text(m, "id");
        if (!waiting.TryGetValue(id, out var w)) return;
        waiting.Remove(id);
        var fly = w.fly;
        var st = w.st;
        fly.pending = false;
        fly.decisions++;
        if (type == "error")
        {
            fly.reason = "뇌 계산 오류: " + MiniJson.Text(m, "error");
            fly.nextAt = Time.time + Cycle;
            if (st.kind == StationKind.Sorter && st.boxPhase == 2) { st.decided = true; st.push = false; st.boxPhase = 3; st.boxT = 0; }
            return;
        }
        fly.values = MiniJson.Obj(m, "values");
        fly.reason = MiniJson.Text(m, "reason");
        var verdict = MiniJson.Obj(m, "verdict");
        fly.outcome = MiniJson.Text(verdict, "outcome");
        var p = MiniJson.Obj(m, "params");
        if (st.fly != fly) return;   // 판단 도중 자리를 바꿨으면 결과만 기록
        switch (st.kind)
        {
            case StationKind.Sorter:
                st.push = MiniJson.Text(verdict, "action") == "push";
                bool right = verdict != null && verdict.TryGetValue("correct", out var c) && c is bool cb && cb;
                if (right) { st.correct++; Earn(st, SortRight); } else { st.wrong++; Earn(st, SortWrong); }
                if (st.push) { st.leverUntil = Time.time + 0.8f; fly.walkUntil = Time.time + 0.8f; }
                st.decided = true;
                if (st.boxPhase == 2) { st.boxPhase = 3; st.boxT = 0; st.boxFrom = st.boxObj.position; }
                break;
            case StationKind.Guard:
                bool alarm = MiniJson.Text(verdict, "action") == "alarm";
                string ev = MiniJson.Text(p, "event");
                if (alarm)
                {
                    st.lampUntil = Time.time + 1.5f;
                    st.gateUntil = Time.time + 2.5f;
                    if (ev == "intruder" && intruderActive && !intruderBlocked)
                    {
                        intruderBlocked = true;
                        st.blocked++;
                        eventText = $"{fly.name}(경비 초소 {st.number})의 거대섬유가 켜져 침입자를 막았어요";
                    }
                    else if (ev != "intruder")
                    {
                        st.falseAlarm++;
                        Earn(st, FalseAlarmLoss);
                    }
                }
                fly.nextAt = Time.time + Cycle;
                break;
            case StationKind.Sugar:
                double speed = MiniJson.Num(verdict, "speed");
                if (speed < 0.02)
                {
                    st.noReaction++;
                    fly.nextAt = Time.time + Cycle;
                }
                else
                {
                    st.cartDur = 6f / (float)speed;
                    st.cartPhase = 1;
                    st.cartT = 0;
                    fly.walkUntil = Time.time + 1f;
                }
                break;
        }
    }

    void Earn(Station st, double v)
    {
        money += v;
        if (st != null) st.earned += v;
        income.Enqueue((Time.time, v));
    }

    // ---------------- 매 프레임 ----------------
    void Update()
    {
        float now = Time.time, dt = Time.deltaTime;
        if (shotAt > 0 && !shotTaken && now >= shotAt && !string.IsNullOrEmpty(shotPath))
        {
            ScreenCapture.CaptureScreenshot(shotPath);
            shotTaken = true;
            Invoke(nameof(QuitNow), 2f);
        }
        while (income.Count > 0 && now - income.Peek().t > 60f) income.Dequeue();

        // 침입자(환경 사건): 평균 20초마다, 4초 안에 경보가 없으면 손실
        if (!intruderActive && now >= nextIntruderAt)
        {
            intruderActive = true;
            intruderBlocked = false;
            intruderStart = now;
            eventText = "침입자 그림자가 다가와요!";
        }
        if (intruderActive && now - intruderStart >= IntruderWindow)
        {
            if (!intruderBlocked)
            {
                var guard = stations.FirstOrDefault(s => s.kind == StationKind.Guard && s.fly != null);
                if (guard != null) guard.missed++;
                Earn(guard, IntruderLoss);
                eventText = guard != null ? "경비 초소의 거대섬유가 문턱을 못 넘어 침입자를 놓쳤어요(−30원)" : "경비 초파리가 없어 침입자에게 설탕을 뺏겼어요(−30원)";
            }
            intruderActive = false;
            nextIntruderAt = now + Exp(IntruderMean);
        }

        foreach (var st in stations)
        {
            if (st.fly != null && st.fly.pending && now - st.fly.pendingSince > 30f) st.fly.pending = false;   // 서버가 끊겼을 때 복구
            switch (st.kind)
            {
                case StationKind.Sorter: TickSorter(st, now, dt); break;
                case StationKind.Guard: TickGuard(st, now); break;
                case StationKind.Sugar: TickSugar(st, now, dt); break;
            }
        }
        foreach (var f in flies)
        {
            if (f.anim == null) continue;
            string want = now < f.walkUntil && f.walkClip != null ? f.walkClip : f.idleClip;
            if (want != null && !f.anim.IsPlaying(want)) f.anim.CrossFade(want, 0.2f);
        }
    }

    void TickSorter(Station st, float now, float dt)
    {
        if (st.lever)
        {
            float k = now < st.leverUntil ? 1f : 0f;
            st.lever.localRotation = Quaternion.Slerp(st.lever.localRotation, st.leverRest * Quaternion.Euler(0, 0, -25f * k), dt * 10f);
        }
        if (st.boxStart == null || st.boxEnd == null) return;
        Vector3 leverPoint = Vector3.Lerp(st.boxStart.position, st.boxEnd.position, 0.3f);
        switch (st.boxPhase)
        {
            case 0:
                if (st.fly == null || st.fly.pending || now < st.fly.nextAt || !brain.Ready) return;
                st.boxKind = env.NextDouble() < 0.5 ? "bright" : "dark";
                st.boxObj = st.boxKind == "bright" ? st.boxBright : st.boxDark;
                if (st.boxObj == null) return;
                st.boxObj.gameObject.SetActive(true);
                st.boxObj.position = st.boxStart.position;
                st.boxFrom = st.boxStart.position;
                st.boxTo = leverPoint;
                st.boxT = 0;
                st.decided = false;
                st.boxPhase = 1;
                Request(st, "sorter", new Dictionary<string, object> { ["box"] = st.boxKind });
                break;
            case 1:
                st.boxT += dt / 1.0f;
                st.boxObj.position = Vector3.Lerp(st.boxFrom, st.boxTo, st.boxT);
                if (st.boxT >= 1f)
                {
                    st.boxPhase = st.decided ? 3 : 2;
                    st.boxT = 0;
                    st.boxFrom = st.boxObj.position;
                }
                break;
            case 3:
                st.boxT += dt / 1.2f;
                Vector3 to = st.push && st.binA ? st.binA.position : st.binB ? st.binB.position : st.boxEnd.position;
                st.boxObj.position = Vector3.Lerp(st.boxFrom, to, Mathf.SmoothStep(0, 1, st.boxT));
                if (st.boxT >= 1f)
                {
                    st.boxObj.gameObject.SetActive(false);
                    st.boxPhase = 0;
                    if (st.fly != null) st.fly.nextAt = now + 0.4f;
                }
                break;
        }
    }

    void TickGuard(Station st, float now)
    {
        if (st.lamp)
        {
            var r = st.lamp.GetComponent<Renderer>();
            if (r != null)
            {
                bool on = now < st.lampUntil && Mathf.Repeat(now * 6f, 1f) < 0.6f;
                r.material.color = on ? new Color(1f, 0.15f, 0.1f) : new Color(0.35f, 0.08f, 0.07f);
                r.material.EnableKeyword("_EMISSION");
                r.material.SetColor("_EmissionColor", on ? new Color(3f, 0.3f, 0.2f) : Color.black);
            }
        }
        // 차단문: Blender 규격 — position.y −20이면 닫혀 받침 윗면에 닿는다
        if (st.gate) st.gate.localPosition = Vector3.Lerp(st.gate.localPosition, st.gateOpen + (now < st.gateUntil ? new Vector3(0, -20f, 0) : Vector3.zero), Time.deltaTime * 6f);
        if (st.intruder)
        {
            bool show = intruderActive && !intruderBlocked;
            st.intruder.gameObject.SetActive(show);
            if (show)
            {
                float k = Mathf.Clamp01((now - intruderStart) / IntruderWindow);
                float grow = Mathf.Lerp(0.5f, 3f, k * k);   // 다가올수록 커지는 그림자(루밍)
                st.intruder.localScale = new Vector3(st.intruderBase.x * grow, st.intruderBase.y, st.intruderBase.z * grow);
            }
        }
        if (st.fly == null || st.fly.pending || now < st.fly.nextAt || !brain.Ready) return;
        string ev = intruderActive && !intruderBlocked ? "intruder" : env.NextDouble() < 0.3 ? "clouds" : "none";
        Request(st, "guard", new Dictionary<string, object> { ["event"] = ev });
    }

    void TickSugar(Station st, float now, float dt)
    {
        if (st.cart == null || st.cartStart == null || st.cartEnd == null) return;
        switch (st.cartPhase)
        {
            case 0:
                st.cart.position = st.cartStart.position;
                if (st.fly == null || st.fly.pending || now < st.fly.nextAt || !brain.Ready) return;
                Earn(st, -sugarHz * SugarCostPerHz);
                Request(st, "sugar", new Dictionary<string, object> { ["sugar_hz"] = (double)sugarHz });
                st.fly.nextAt = now + 0.2f;
                break;
            case 1:
                st.cartT += dt / st.cartDur;
                st.cart.position = Vector3.Lerp(st.cartStart.position, st.cartEnd.position, st.cartT);
                if (st.cartT >= 1f) { st.deliveries++; Earn(st, DeliveryPay); st.cartPhase = 2; st.cartT = 0; }
                break;
            case 2:
                st.cartT += dt / 1.2f;
                st.cart.position = Vector3.Lerp(st.cartEnd.position, st.cartStart.position, st.cartT);
                if (st.cartT >= 1f) { st.cartPhase = 0; if (st.fly != null) st.fly.nextAt = now + 0.3f; }
                break;
        }
    }

    void QuitNow() => Application.Quit();

    // ---------------- 화면 ----------------
    double FlyCost => Math.Round(60 * Math.Pow(1.3, fliesBought));
    double StationCost => Math.Round(100 * Math.Pow(1.4, stationsBought));

    void Styles()
    {
        if (font != null) return;
        font = Font.CreateDynamicFontFromOSFont(new[] { "Apple SD Gothic Neo", "AppleGothic", "Arial Unicode MS" }, 15);
        GUI.skin.font = font;
        title = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold, richText = true };
        body = new GUIStyle(GUI.skin.label) { fontSize = 15, wordWrap = true, richText = true };
        small = new GUIStyle(GUI.skin.label) { fontSize = 13, wordWrap = true, richText = true };
    }

    void Bar(Rect r, string label, double v, double thr, double max, Color c)
    {
        GUI.Label(new Rect(r.x, r.y, 110, r.height), label, small);
        var track = new Rect(r.x + 112, r.y + 5, r.width - 210, r.height - 10);
        GUI.color = new Color(0, 0, 0, 0.45f);
        GUI.DrawTexture(track, Texture2D.whiteTexture);
        GUI.color = c;
        GUI.DrawTexture(new Rect(track.x, track.y, track.width * Mathf.Clamp01((float)(v / max)), track.height), Texture2D.whiteTexture);
        GUI.color = Color.white;
        GUI.DrawTexture(new Rect(track.x + track.width * Mathf.Clamp01((float)(thr / max)) - 1, track.y - 3, 2, track.height + 6), Texture2D.whiteTexture);
        GUI.Label(new Rect(track.xMax + 8, r.y, 100, r.height), $"{v:0.#} / {thr:0.#}Hz", small);
    }

    void OnGUI()
    {
        Styles();
        GUI.color = Color.white;
        double perMin = income.Sum(x => x.v);

        GUILayout.BeginArea(new Rect(14, 14, 460, 420), GUI.skin.box);
        GUILayout.Label("초파리 공장", title);
        GUILayout.Label($"돈 <b>{money:0}원</b>   최근 1분 {perMin:+0;-0;0}원", body);
        GUILayout.Label(brain.Connected ? $"뇌 서버 연결됨 · 계산 프로세스 {brain.WorkersReady}/{brain.Workers} · 대기열 {brain.Queue}" : "뇌 서버 연결 기다리는 중(factory/server/brain_server.py)", small);
        GUILayout.Label("초파리는 조종도 학습도 되지 않아요. 타고난 반사 뉴런(다가가기·도주·섭식) 발화만으로 일해요.", small);
        GUILayout.Label($"<i>{eventText}</i>", small);
        GUILayout.Space(6);
        if (GUILayout.Button($"초파리 들이기 {FlyCost}원") && money >= FlyCost)
        {
            money -= FlyCost;
            fliesBought++;
            AddFly(stations.FirstOrDefault(s => s.fly == null));
        }
        GUILayout.BeginHorizontal();
        foreach (var (kind, label) in new[] { (StationKind.Sorter, "분류대"), (StationKind.Guard, "경비 초소"), (StationKind.Sugar, "설탕대") })
        {
            if (GUILayout.Button($"{label} {StationCost}원") && money >= StationCost)
            {
                money -= StationCost;
                stationsBought++;
                var st = AddStation(kind);
                var idle = flies.FirstOrDefault(f => f.station == null);
                if (idle != null) Assign(idle, st);
                PlaceFlies();
            }
        }
        GUILayout.EndHorizontal();
        GUILayout.Label($"설탕 농도(당 감각뉴런 자극) {sugarHz:0}Hz · 한 번에 {sugarHz * SugarCostPerHz:0.0}원", small);
        sugarHz = Mathf.Round(GUILayout.HorizontalSlider(sugarHz, 50f, 200f) / 10f) * 10f;
        var idleFlies = flies.Where(f => f.station == null).ToList();
        if (idleFlies.Count > 0) GUILayout.Label("쉬는 초파리: " + string.Join(", ", idleFlies.Select(f => f.name)), small);
        GUILayout.EndArea();

        float w = 520, x = Screen.width - w - 14;
        GUILayout.BeginArea(new Rect(x, 14, w, Screen.height - 28), GUI.skin.box);
        scroll = GUILayout.BeginScrollView(scroll);
        foreach (var st in stations)
        {
            var f = st.fly;
            GUILayout.Label($"<b>{st.Name}</b> — {(f != null ? $"{f.name} (개체 {f.seed})" : "초파리 없음")}", body);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("초파리 바꾸기", GUILayout.Width(120))) Swap(st);
            if (st.kind == StationKind.Sorter && f != null)
            {
                if (GUILayout.Button("레버 감도 −", GUILayout.Width(100))) f.leverThreshold += 0.5;
                if (GUILayout.Button("레버 감도 +", GUILayout.Width(100))) f.leverThreshold = Math.Max(0.5, f.leverThreshold - 0.5);
            }
            GUILayout.EndHorizontal();
            string stats = st.kind == StationKind.Sorter
                ? $"맞음 {st.correct} · 틀림 {st.wrong} · 정확도 {(st.correct + st.wrong > 0 ? 100.0 * st.correct / (st.correct + st.wrong) : 0):0}% · 벌이 {st.earned:+0;-0;0}원"
                : st.kind == StationKind.Guard
                    ? $"막음 {st.blocked} · 놓침 {st.missed} · 헛경보 {st.falseAlarm} · 손익 {st.earned:+0;-0;0}원"
                    : $"배달 {st.deliveries} · 반응 없음 {st.noReaction} · 손익 {st.earned:+0;-0;0}원";
            GUILayout.Label(stats, small);
            if (f != null)
            {
                Rect r = GUILayoutUtility.GetRect(w - 40, 24);
                var v = f.values;
                if (st.kind == StationKind.Sorter) Bar(r, "다가가기", MiniJson.Num(v, "approach_hz"), f.leverThreshold, 40, new Color(0.9f, 0.45f, 0.3f));
                else if (st.kind == StationKind.Guard) Bar(r, "도주 거대섬유", MiniJson.Num(v, "GF_peak50ms_hz"), GfThreshold, 200, new Color(0.35f, 0.6f, 0.95f));
                else Bar(r, "섭식 MN9", MiniJson.Num(v, "MN9_mean_hz"), Mn9Ref, 90, new Color(0.55f, 0.8f, 0.45f));
                GUILayout.Label((f.pending ? "뇌 계산 중… " : "왜: ") + f.reason + (f.outcome != "" ? $" → {f.outcome}" : ""), small);
                GUILayout.Label(f.aptText, small);
            }
            GUILayout.Space(10);
        }
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }

    void Swap(Station st)
    {
        var idle = flies.FirstOrDefault(f => f.station == null);
        if (idle != null)
        {
            Assign(idle, st);
            return;
        }
        if (st.fly == null) return;
        int i = stations.IndexOf(st);
        var other = stations.Skip(i + 1).Concat(stations.Take(i)).FirstOrDefault(s => s.fly != null);
        if (other == null) return;
        var a = st.fly;
        var b = other.fly;
        a.station = other;
        other.fly = a;
        b.station = st;
        st.fly = b;
        PlaceFlies();
    }
}
