using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

/// <summary>
/// 초파리 뇌로 움직이는 공장(방치형) — 게임 규칙·경제·화면.
///
/// 원칙: 초파리는 조종도 학습도 되지 않는다. 일은 Shiu 2024 전뇌 모델에서 재현되는 타고난 반사로만 한다.
///   불빛 분류대 = 다가가기 명령 뉴런(oDN1·P9) / 경비 초소 = 거대섬유 도주(DNp01) / 설탕 배달대 = 당 → 섭식 운동뉴런(MN9).
///   판정은 매번 뇌 서버의 진짜 계산 결과다. 여기서 쓰는 난수는 환경(어떤 상자가 오나·침입자가 언제 오나)뿐이다.
/// 몸짓: 걷기·몸단장·날갯짓 모양은 Blender 클립(body/초파리.fbx)이고, 언제·얼마나 빨리·어떤 동작을 할지는 뇌 판정이 정한다.
///   분류대 — 다가가기가 문턱을 넘으면 레버 쪽으로 몸을 민다 / 설탕대 — 설탕 알갱이를 지고 MN9 속도로 수레를 끌고 걷는다 /
///   경비 — 거대섬유가 켜지면 날갯짓하며 뛰어오른다(실제 거대섬유 도약 도주).
/// 플레이어가 하는 일: 초파리를 어느 작업대에 둘지, 작업대·초파리 구입, 설탕 농도, 분류대 레버 감도.
/// 경제 수치 근거: factory/brain/reflex_probe.json 실측.
/// </summary>
public enum StationKind { Sorter, Guard, Sugar }

public partial class Fly
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
    public string idleClip, walkClip, flyClip;
    public Vector3 home;
    public Quaternion homeRot;
    public float pushUntil, hopStart = -10f;
    public float walkSpeed = 1f;
    public Transform carry;
    public Vector3 forwardModel = Vector3.right;   // 모델 머리 방향(뼈에서 읽음)
    public int decisions;
    // 장비(갑옷) 단계 0~3 — 몸 동작만 빠르게 한다. 레버를 밀지·뛸지·수레 속도 비율은 그대로 뇌 판정이다.
    public int gear;
    public float BodySpeed => 1f + 0.35f * gear;
    public float pushDur = 0.9f;
    public readonly List<(int tier, GameObject go)> armor = new List<(int, GameObject)>();
}

public partial class Station
{
    public StationKind kind;
    public int number;
    public Fly fly;
    public GameObject root;
    public Transform flySpot, lever, boxBright, boxDark, boxStart, boxEnd, binA, binB, lamp, gate, intruder, cart, cartStart, cartEnd, leverHandle;
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
public partial class FactoryGame : MonoBehaviour
{
    const double SortRight = 4, SortWrong = -2, IntruderLoss = -30, FalseAlarmLoss = -3, DeliveryPay = 10, SugarCostPerHz = 0.02;
    const float Cycle = 2.5f, CellX = 380f, CellZ = 280f, IntruderWindow = 4f, IntruderMean = 20f, GfThreshold = 60f, Mn9Ref = 71f;

    public double money = 200;
    public float sugarHz = 150f;

    readonly List<Fly> flies = new List<Fly>();
    readonly List<Station> stations = new List<Station>();
    readonly Dictionary<string, (Fly fly, Station st)> waiting = new Dictionary<string, (Fly, Station)>();
    readonly Queue<(float t, double v)> income = new Queue<(float, double)>();
    readonly System.Random env = new System.Random(20260913);
    readonly string[] names = { "누리", "보리", "호박", "초코", "깨비", "망고", "두부", "율무", "콩이", "모카", "단지", "쑥이" };

    BrainClient brain;
    int nextSeed = 101, requestN, stationsBought;
    bool testGear;
    static readonly int[] GearCost = { 80, 180, 400 };
    static readonly string[] GearName = { "맨몸", "가죽 조끼", "쇠 판금", "황금 갑옷" };
    bool intruderActive, intruderBlocked;
    float intruderStart, nextIntruderAt;
    string eventText = "공장 가동 준비 중";
    Camera cam;
    Font font;
    GUIStyle title, body, small;
    Vector2 scroll;
    Transform worldRoot;
    int cols = 1;
    float roomHalf = 500f;   // 정사각형 방 절반 길이(바닥 타일 100 단위)

    // 카메라: -1 = 공장 전체, 0.. = 작업대 확대
    int focus = -1;
    float camYaw = -90f, camPitch = 32f, camDist = 700f, yawNow = -90f;
    Vector3 camTarget;
    bool camInit;

    // 캡처(검증용): -shotDir 폴더 -shotPlan "45:all,52:s0,..."  또는 -shot 파일 -shotAfter 초
    string shotPath, shotDir;
    float shotAt = -1f;
    bool shotTaken;
    readonly List<(float at, string view)> shotPlan = new List<(float, string)>();
    int shotIndex;
    readonly HashSet<string> eventShotsDone = new HashSet<string>();
    string eventShotName;
    float eventShotAt = -1f, planDoneAt = -1f;

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
        if (testGear)
        {
            // 검증 캡처용: 분류대 1·2 = 가죽·쇠, 경비·설탕 = 황금
            int[] g = { 1, 2, 3, 3 };
            for (int i = 0; i < 4; i++) { flies[i].gear = g[i]; ApplyGear(flies[i]); }
        }
        StartDefense();
        nextIntruderAt = Time.time + Exp(IntruderMean);
        Invoke(nameof(LogScene), 3f);
    }

    void ParseArgs()
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "-shot") shotPath = args[i + 1];
            if (args[i] == "-shotAfter" && float.TryParse(args[i + 1], out float s)) shotAt = s;
            if (args[i] == "-shotDir") shotDir = args[i + 1];
            if (args[i] == "-testGear") testGear = args[i + 1] == "1";
            if (args[i] == "-testSoldiers") testSoldiers = args[i + 1] == "1";
            if (args[i] == "-testFactoryHp" && double.TryParse(args[i + 1], out double hp)) factoryHp = hp;
            if (args[i] == "-shotPlan")
                foreach (var part in args[i + 1].Split(','))
                {
                    var kv = part.Split(':');
                    if (kv.Length == 2 && float.TryParse(kv[0], out float at)) shotPlan.Add((at, kv[1]));
                }
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
        cam.backgroundColor = new Color(0.035f, 0.037f, 0.04f);
        cam.fieldOfView = 40f;
        cam.farClipPlane = 8000f;
        cam.nearClipPlane = 0.5f;

        var sun = new GameObject("햇빛").AddComponent<Light>();
        sun.type = LightType.Directional;
        sun.intensity = 0.32f;                       // 창 없는 공장: 약한 푸른 회색 채움빛 + 작업대마다 매단 전등(RebuildFloor)
        sun.color = new Color(0.72f, 0.78f, 0.88f);
        sun.shadows = LightShadows.Soft;
        sun.transform.rotation = Quaternion.Euler(52f, -120f, 0f);
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.10f, 0.105f, 0.115f);
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Linear;
        RenderSettings.fogColor = new Color(0.035f, 0.037f, 0.04f);
        RenderSettings.fogStartDistance = 2200f;
        RenderSettings.fogEndDistance = 4800f;
        QualitySettings.pixelLightCount = 16;

        var under = GameObject.CreatePrimitive(PrimitiveType.Plane);
        under.name = "바닥_바탕";
        under.transform.localScale = new Vector3(600f, 1f, 600f);
        under.transform.position = new Vector3(0f, -2.5f, 0f);
        under.GetComponent<Renderer>().material.color = new Color(0.04f, 0.04f, 0.045f);
        worldRoot = new GameObject("공장 바닥·벽").transform;
        RebuildFloor();
    }

    /// <summary>정사각형 방: 작업대를 ⌈√n⌉열 격자로 놓고, 그 둘레에 바닥을 깔고 벽을 두른다.</summary>
    void Relayout()
    {
        int n = stations.Count;
        cols = Mathf.Max(1, Mathf.CeilToInt(Mathf.Sqrt(n)));
        int rows = Mathf.Max(1, Mathf.CeilToInt(n / (float)cols));
        for (int i = 0; i < n; i++)
        {
            int c = i % cols, r = i / cols;
            stations[i].root.transform.position = new Vector3((c - (cols - 1) / 2f) * CellX, 0f, (r - (rows - 1) / 2f) * CellZ);
        }
        roomHalf = Mathf.Ceil((Mathf.Max(cols * CellX, rows * CellZ) + 200f) / 200f) * 100f;
        if (worldRoot != null) RebuildFloor();
        PlaceFlies();
    }

    void RebuildFloor()
    {
        // Blender 바닥 타일(100×100, 이어도 무늬 안 끊김)·벽 한 칸(폭 100, 높이 60)을 방 크기에 맞춰 깐다.
        foreach (Transform c in worldRoot) Destroy(c.gameObject);
        var tile = Resources.Load<GameObject>("Models/floor_concrete") ?? Resources.Load<GameObject>("Models/floor_tile");
        var wall = Resources.Load<GameObject>("Models/wall_steel") ?? Resources.Load<GameObject>("Models/wall");
        var pillar = Resources.Load<GameObject>("Models/pillar_h");
        var lamp = Resources.Load<GameObject>("Models/lamp_hanging");
        Vector3 tileScale = Vector3.one, wallScale = Vector3.one;
        if (tile)
        {
            var probe = Instantiate(tile);
            FitWidth(probe, 100f, "바닥 타일");
            tileScale = probe.transform.localScale;
            Destroy(probe);
        }
        if (wall)
        {
            var probe = Instantiate(wall);
            FitWidth(probe, 100f, "벽");
            wallScale = probe.transform.localScale;
            Destroy(probe);
        }
        int tiles = Mathf.RoundToInt(roomHalf * 2f / 100f);
        for (int ix = 0; ix < tiles; ix++)
            for (int iz = 0; iz < tiles; iz++)
                if (tile) Instantiate(tile, new Vector3(-roomHalf + ix * 100f + 50f, 0f, -roomHalf + iz * 100f + 50f), Quaternion.identity, worldRoot).transform.localScale = tileScale;
        if (wall)
            for (int k = 0; k < tiles; k++)
            {
                float t = -roomHalf + k * 100f + 50f;
                if (Mathf.Abs(t) < GateHalf) continue;   // 사방 문 자리(FactoryGame.Defense.cs)
                // 카메라 반대쪽 두 벽(−X·−Z)은 온전히, 카메라 쪽 두 벽(+X·+Z)은 낮은 턱 — 방 안이 가려지지 않게
                PlaceWall(wall, new Vector3(-roomHalf, 0f, t), 90f, wallScale, false);
                PlaceWall(wall, new Vector3(t, 0f, -roomHalf), 0f, wallScale, false);
                PlaceWall(wall, new Vector3(roomHalf, 0f, t), 90f, wallScale, true);
                PlaceWall(wall, new Vector3(t, 0f, roomHalf), 0f, wallScale, true);
            }
        if (pillar)
            foreach (var at in new[] { new Vector3(-roomHalf, 0, -roomHalf), new Vector3(-roomHalf, 0, roomHalf), new Vector3(roomHalf, 0, -roomHalf),
                                       new Vector3(-roomHalf, 0, -GateHalf - 12), new Vector3(-roomHalf, 0, GateHalf + 12),
                                       new Vector3(-GateHalf - 12, 0, -roomHalf), new Vector3(GateHalf + 12, 0, -roomHalf) })
            {
                var p = Instantiate(pillar, at, Quaternion.identity, worldRoot);
                FitWidth(p, 20f, "기둥");
            }
        foreach (var st in stations)
        {
            // 작업대마다 매단 공장 전등 + 아래로 비추는 따뜻한 스포트라이트
            Vector3 c = BoundsOf(st.root).center;
            Vector3 top = new Vector3(c.x, 150f, c.z);
            if (lamp)
            {
                var l = Instantiate(lamp, top, Quaternion.identity, worldRoot);
                FitWidth(l, 32f, "전등");
            }
            var spot = new GameObject("전등빛").AddComponent<Light>();
            spot.transform.SetParent(worldRoot, false);
            spot.transform.position = top + Vector3.up * 6f;
            spot.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            spot.type = LightType.Spot;
            spot.range = 420f;
            spot.spotAngle = 88f;
            spot.intensity = 2.6f;
            spot.color = new Color(1f, 0.86f, 0.66f);
            spot.shadows = LightShadows.Soft;
        }
        BuildYardAndGates(tileScale);
        DressMaterials(worldRoot.gameObject);
    }

    void PlaceWall(GameObject prefab, Vector3 pos, float yaw, Vector3 scale, bool low)
    {
        var w = Instantiate(prefab, pos, Quaternion.Euler(0f, yaw, 0f), worldRoot);
        w.transform.localScale = low ? new Vector3(scale.x, scale.y * 0.2f, scale.z) : scale;
    }

    // ---------------- 크기·진단 ----------------
    /// <summary>
    /// FBX 단위 해석이 어긋나 모델이 너무 크거나 작으면 blender 세션 규격 크기(가로 X)에 맞춘다.
    /// 규격: 분류대 243.4 · 경비 초소 72 · 설탕대 165.5 · 바닥 타일 100 · 벽 100 · 초파리(×10) 약 42.6.
    /// </summary>
    static void FitWidth(GameObject go, float expectedX, string label)
    {
        var b = BoundsOf(go);
        float actual = Mathf.Max(b.size.x, b.size.z);
        if (actual < 1e-4f) return;
        float ratio = expectedX / actual;
        if (ratio > 0.5f && ratio < 2f) return;
        go.transform.localScale *= ratio;
        Debug.Log($"SCENE_RESCALE {label} 실제 {actual} → 규격 {expectedX} (×{ratio})");
    }

    /// <summary>
    /// 초파리 FBX 안의 메시·뼈대 오브젝트에 단위 변환 배율 ×100이 들어 있고 메시 꼭짓점은 mm(가로 4.26)다(2026-09-13 진단 FLY_DIAG).
    /// 그래서 화면 크기 = 메시 가로 × 뼈대 누적 배율. renderer.bounds나 BakeMesh는 이 배율을 제대로 안 담아 쓰지 않는다.
    /// </summary>
    void FitFlyWidth(GameObject go, float expected)
    {
        var smr = go.GetComponentInChildren<SkinnedMeshRenderer>();
        if (smr == null || smr.sharedMesh == null) return;
        smr.updateWhenOffscreen = true;
        Transform unitOf = smr.rootBone && smr.rootBone.parent ? smr.rootBone.parent : smr.transform;
        float inner = unitOf.lossyScale.x / go.transform.lossyScale.x;   // 루트 아래 누적 배율(×100)
        Vector3 m = smr.sharedMesh.bounds.size;
        float native = Mathf.Max(m.x, m.y, m.z) * inner;
        if (native > 1e-4f) go.transform.localScale = Vector3.one * (expected / native);
        Debug.Log($"SCENE_FLY_FIT 메시 {m} × 안쪽 배율 {inner} = {native} → 규격 {expected}, 루트 배율 {go.transform.localScale.x}");
    }

    // Blender 재질(gen_factory.py MATS)은 노드 텍스처라 FBX에 이미지·색이 안 실려 온다(모두 흰색·텍스처없음) → 이름으로 다시 입힌다.
    static readonly Dictionary<string, (string tex, Color linear, float metal, float rough)> MatSpec = new Dictionary<string, (string, Color, float, float)>
    {
        ["공장_금속"] = ("factory_metal", Color.white, 0.85f, 0.42f),
        ["공장_금속_짙음"] = (null, new Color(0.05f, 0.05f, 0.055f), 0.7f, 0.5f),
        ["공장_나무"] = ("factory_wood", Color.white, 0f, 0.72f),
        ["공장_흰플라스틱"] = ("factory_plastic", Color.white, 0f, 0.45f),
        ["공장_고무벨트"] = ("factory_rubber", Color.white, 0f, 0.9f),
        ["공장_손잡이"] = (null, new Color(0.42f, 0.06f, 0.03f), 0f, 0.35f),
        ["공장_바닥"] = ("factory_floor", Color.white, 0f, 0.7f),
        ["공장_벽"] = ("factory_plaster", Color.white, 0f, 0.9f),
        ["공장_설탕"] = (null, new Color(0.92f, 0.92f, 0.9f), 0f, 0.25f),
        ["공장_도자기"] = (null, new Color(0.80f, 0.80f, 0.77f), 0f, 0.18f),
        // factory/blender/gen_room.py — 어두운 공장
        ["공장_콘크리트"] = ("factory_concrete", Color.white, 0f, 0.9f),
        ["공장_강철벽"] = ("factory_steelwall", Color.white, 0.6f, 0.6f),
        ["공장_경고띠"] = ("factory_hazard", Color.white, 0f, 0.7f),
        ["공장_철골"] = (null, new Color(0.10f, 0.105f, 0.11f), 0.7f, 0.55f),
        ["공장_전등갓"] = (null, new Color(0.12f, 0.15f, 0.14f), 0.5f, 0.4f),
        ["공장_전구"] = (null, new Color(1.0f, 0.86f, 0.6f), 0f, 0.2f),
        // factory/blender/gen_armor.py — 초파리 장비
        ["갑옷_가죽"] = (null, new Color(0.30f, 0.15f, 0.06f), 0f, 0.8f),
        ["갑옷_가죽_테"] = (null, new Color(0.12f, 0.06f, 0.03f), 0f, 0.9f),
        ["갑옷_쇠"] = (null, new Color(0.56f, 0.57f, 0.60f), 1f, 0.35f),
        ["갑옷_쇠_테"] = (null, new Color(0.20f, 0.20f, 0.22f), 1f, 0.45f),
        ["갑옷_금"] = (null, new Color(0.85f, 0.62f, 0.16f), 1f, 0.25f),
        ["갑옷_금_테"] = (null, new Color(0.55f, 0.33f, 0.06f), 1f, 0.3f),
        ["갑옷_볏"] = (null, new Color(0.62f, 0.03f, 0.03f), 0f, 0.6f),
    };
    static readonly HashSet<Material> dressed = new HashSet<Material>();

    static void DressMaterials(GameObject go)
    {
        foreach (var r in go.GetComponentsInChildren<Renderer>())
            foreach (var m in r.sharedMaterials)
            {
                if (m == null || dressed.Contains(m)) continue;
                dressed.Add(m);
                string key = m.name.Replace(" (Instance)", "");
                int dot = key.IndexOf('.');
                if (dot > 0) key = key.Substring(0, dot);
                if (!MatSpec.TryGetValue(key, out var spec)) continue;
                m.color = spec.linear.gamma;
                if (spec.tex != null) m.mainTexture = Resources.Load<Texture2D>("Models/Textures/" + spec.tex);
                if (m.HasProperty("_Metallic")) m.SetFloat("_Metallic", spec.metal * 0.35f);   // 반사 프로브가 없어 금속 그대로면 검게 보인다
                if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", 1f - spec.rough);
                if (key == "공장_전구" || key.EndsWith("눈"))                   // 전구·괴물 눈은 스스로 빛난다
                {
                    m.EnableKeyword("_EMISSION");
                    m.SetColor("_EmissionColor", (key == "공장_전구" ? new Color(1f, 0.82f, 0.55f) : spec.linear) * 2.5f);
                }
            }
    }

    static Bounds BoundsOf(GameObject go)
    {
        var rs = go.GetComponentsInChildren<Renderer>();
        if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
        var b = rs[0].bounds;
        foreach (var r in rs) b.Encapsulate(r.bounds);
        return b;
    }

    static string Materials(GameObject go) => string.Join(" | ", go.GetComponentsInChildren<Renderer>().Take(4)
        .SelectMany(r => r.sharedMaterials).Where(m => m != null)
        .Select(m => $"{m.name}:{(m.mainTexture ? m.mainTexture.name : "텍스처없음")}:{m.color}"));

    void LogScene()
    {
        Debug.Log($"SCENE_CAM pos={cam.transform.position} fwd={cam.transform.forward}");
        foreach (var st in stations)
        {
            var b = BoundsOf(st.root);
            Debug.Log($"SCENE_STATION {st.Name} root={st.root.transform.position} scale={st.root.transform.lossyScale} size={b.size} flySpot={(st.flySpot ? st.flySpot.position.ToString() : "없음")} mats={Materials(st.root)}");
        }
        foreach (var f in flies)
        {
            var b = BoundsOf(f.go);
            Debug.Log($"SCENE_FLY {f.name} pos={f.go.transform.position} size={b.size} fwdModel={f.forwardModel} clips={f.idleClip},{f.walkClip},{f.flyClip} mats={Materials(f.go)}");
        }
    }

    // ---------------- 작업대 ----------------
    Station AddStation(StationKind kind)
    {
        var st = new Station { kind = kind, number = stations.Count(s => s.kind == kind) + 1 };
        string asset = kind == StationKind.Sorter ? "station_sorter" : kind == StationKind.Guard ? "station_guard" : "station_sugar";
        var prefab = Resources.Load<GameObject>("Models/" + asset);
        st.root = prefab != null ? Instantiate(prefab) : Fallback(kind);
        st.root.name = st.Name;
        if (prefab != null)
        {
            FitWidth(st.root, kind == StationKind.Sorter ? 243.4f : kind == StationKind.Guard ? 72f : 165.5f, st.Name);
            DressMaterials(st.root);
        }
        stations.Add(st);
        Transform T(string n) => FindDeep(st.root.transform, n);
        st.flySpot = T("초파리_자리");
        st.lever = T("레버");
        st.leverHandle = T("레버_손잡이");
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
        Relayout();
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
        // 설비 FBX가 없을 때 쓰는 기본 도형. 소켓 이름은 FBX와 같다.
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
            lever.transform.localPosition = new Vector3(-90, 4.5f, 14);
            Box("레버_막대", new Vector3(0, 6, 0), new Vector3(2, 12, 2), metal, lever.transform);
            Box("통_A", new Vector3(-58, 7, -42), new Vector3(30, 14, 30), wood);
            Box("통_B", new Vector3(100, 7, 0), new Vector3(30, 14, 44), wood);
            Box("상자_밝음", new Vector3(-65, 24, 0), new Vector3(18, 18, 18), new Color(0.92f, 0.92f, 0.92f));
            Box("상자_어두움", new Vector3(-65, 24, 0), new Vector3(18, 18, 18), new Color(0.08f, 0.08f, 0.08f));
            Socket("초파리_자리", new Vector3(-108, 1.5f, 0));
            Socket("레버_손잡이", new Vector3(-90, 12.5f, 14));
            Socket("상자_시작", new Vector3(-65, 15, 0));
            Socket("상자_끝", new Vector3(65, 15, 0));
            Socket("통_A_입구", new Vector3(-58, 14, -42));
            Socket("통_B_입구", new Vector3(100, 14, 0));
        }
        else if (kind == StationKind.Guard)
        {
            Box("받침", new Vector3(0, 1.5f, 0), new Vector3(60, 3, 60), wood);
            Box("경보등", new Vector3(31, 48, 0), new Vector3(10, 8, 10), new Color(0.5f, 0.1f, 0.1f));
            Box("차단문", new Vector3(34.7f, 23, 0), new Vector3(4, 40, 50), metal);
            var disk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disk.name = "침입자";
            disk.transform.SetParent(root.transform, false);
            disk.transform.localPosition = new Vector3(0, 80, 0);
            disk.transform.localScale = new Vector3(20, 0.5f, 20);
            disk.GetComponent<Renderer>().material.color = new Color(0.05f, 0.05f, 0.06f);
            Socket("초파리_자리", new Vector3(-8, 3, 0));
        }
        else
        {
            var dish = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            dish.name = "설탕";
            dish.transform.SetParent(root.transform, false);
            dish.transform.localPosition = new Vector3(-50, 1.5f, 0);
            dish.transform.localScale = new Vector3(20, 1.5f, 20);
            dish.GetComponent<Renderer>().material.color = new Color(0.95f, 0.95f, 0.92f);
            Box("레일", new Vector3(20, 0.75f, 0), new Vector3(120, 1.5f, 10), metal);
            Box("선반", new Vector3(96, 6, 0), new Vector3(24, 12, 36), wood);
            Box("수레", new Vector3(-30, 6, 0), new Vector3(20, 10, 16), new Color(0.35f, 0.5f, 0.7f));
            Socket("초파리_자리", new Vector3(-70, 0, 0));
            Socket("수레_시작", new Vector3(-30, 1.5f, 0));
            Socket("수레_끝", new Vector3(70, 1.5f, 0));
        }
        return root;
    }

    // ---------------- 초파리 ----------------
    Fly AddFly(Station st, bool soldier = false)
    {
        var f = new Fly { seed = nextSeed++, name = names[flies.Count % names.Length], soldier = soldier };
        flies.Add(f);
        var prefab = Resources.Load<GameObject>("Models/초파리_장비") ?? Resources.Load<GameObject>("Models/초파리");
        if (prefab != null)
        {
            f.go = Instantiate(prefab);
            FitFlyWidth(f.go, 42.6f);   // 공장 장면은 초파리 ×10(몸 FBX 1 = 1mm) — 규격 가로 약 42.6
        }
        else
        {
            f.go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            f.go.transform.localScale = new Vector3(10, 6, 10);
        }
        f.go.name = $"초파리 {f.name}";
        var head = FindDeep(f.go.transform, "Head");
        var thorax = FindDeep(f.go.transform, "Thorax");
        if (head && thorax)
        {
            Vector3 fwd = head.position - thorax.position;
            fwd.y = 0;
            if (fwd.sqrMagnitude > 1e-8f) f.forwardModel = fwd.normalized;
        }
        foreach (var t in f.go.GetComponentsInChildren<Transform>(true))
            if (t.name.StartsWith("갑옷") && t.name.Length > 2 && char.IsDigit(t.name[2]))
                f.armor.Add((t.name[2] - '0', t.gameObject));
            else if (t.name.StartsWith("병정_")) f.soldierGear.Add(t.gameObject);
        DressMaterials(f.go);
        ApplyGear(f);
        f.anim = f.go.GetComponentInChildren<Animation>();
        if (f.anim != null)
        {
            foreach (AnimationState s in f.anim)
            {
                if (s.name.Contains("Idle_Groom")) f.idleClip = s.name;
                if (s.name.Contains("Walk_Tripod")) f.walkClip = s.name;
                if (s.name.Contains("Flight_Wingbeat")) f.flyClip = s.name;
                s.wrapMode = WrapMode.Loop;
            }
            if (f.idleClip != null) f.anim.Play(f.idleClip);
        }
        // 설탕 알갱이(설탕대에서 지고 걷는다)
        var grain = GameObject.CreatePrimitive(PrimitiveType.Cube);
        grain.name = "지고 가는 설탕";
        Destroy(grain.GetComponent<Collider>());
        grain.GetComponent<Renderer>().material.color = new Color(0.98f, 0.97f, 0.92f);
        grain.transform.localScale = Vector3.one * 5f;
        grain.SetActive(false);
        f.carry = grain.transform;
        Assign(f, st);
        if (soldier)
        {
            f.aptDone = true;
            f.aptText = "병정 초파리 — 작업대 적성 검사 없음";
        }
        else SendAptitude(f);
        return f;
    }

    static void ApplyGear(Fly f)
    {
        foreach (var (tier, go) in f.armor) go.SetActive(tier == f.gear);
        ApplySoldierGear(f);
    }

    void SendAptitude(Fly f)
    {
        f.aptDone = false;
        f.aptText = "적성 검사 중(뇌 계산 12번)";
        if (brain != null && brain.Connected)
            brain.Send(new Dictionary<string, object> { ["type"] = "aptitude", ["id"] = $"apt{f.seed}", ["fly_seed"] = (double)f.seed });
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
            if (f.soldier) continue;
            Vector3 spot, target;
            if (f.station != null)
            {
                var st = f.station;
                spot = st.flySpot ? st.flySpot.position : st.root.transform.position + new Vector3(-80, 0, 0);
                target = st.kind == StationKind.Sorter && st.leverHandle ? st.leverHandle.position
                    : st.kind == StationKind.Sugar && st.cartStart ? st.cartStart.position : st.root.transform.position;
            }
            else
            {
                spot = new Vector3(-roomHalf + 60f, 0f, -roomHalf + 60f + idle * 55f);   // 방 안쪽 구석 쉼터
                target = spot + new Vector3(60, 0, 0);
                idle++;
            }
            f.home = spot;
            f.homeRot = FaceRotation(f, spot, target);
            f.go.transform.position = spot;
            f.go.transform.rotation = f.homeRot;
        }
    }

    static Quaternion FaceRotation(Fly f, Vector3 from, Vector3 to)
    {
        // 모델 머리 방향(뼈 Thorax → Head)이 to를 보게 도는 회전(FBX 축 변환에 기대지 않음)
        Vector3 want = to - from;
        want.y = 0;
        if (want.sqrMagnitude < 1e-6f) return Quaternion.identity;
        return Quaternion.FromToRotation(f.forwardModel, want.normalized);
    }

    void PlayClip(Fly f, string clip, float speed = 1f)
    {
        if (f.anim == null || clip == null) return;
        var state = f.anim[clip];
        if (state != null) state.speed = speed;
        if (!f.anim.IsPlaying(clip)) f.anim.CrossFade(clip, 0.2f);
    }

    void UpdateFlies(float now, float dt)
    {
        foreach (var f in flies)
        {
            if (f.soldier) continue;   // 병정은 UpdateSoldier
            var st = f.station;
            Vector3 pos = f.home;
            Quaternion rot = f.homeRot;
            string clip = f.idleClip;
            float clipSpeed = 1f;
            bool carrying = false;
            if (st != null)
            {
                switch (st.kind)
                {
                    case StationKind.Sorter:
                        if (now < f.pushUntil && st.leverHandle)
                        {
                            // 다가가기 뉴런 판정 → 레버 쪽으로 몸을 밀고 돌아옴
                            float k = 1f - Mathf.Abs((f.pushUntil - now) / (f.pushDur * 0.5f) - 1f);
                            Vector3 toLever = st.leverHandle.position - f.home;
                            toLever.y = 0;
                            pos = f.home + toLever.normalized * Mathf.Min(toLever.magnitude * 0.6f, 14f) * Mathf.Clamp01(k);
                            clip = f.walkClip;
                            clipSpeed = 2f * f.BodySpeed;
                        }
                        break;
                    case StationKind.Guard:
                        float since = now - f.hopStart;
                        if (since < 0.9f)
                        {
                            // 거대섬유 판정 → 날갯짓하며 도약
                            float k = since / 0.9f;
                            pos = f.home + Vector3.up * Mathf.Sin(Mathf.PI * k) * 35f;
                            clip = f.flyClip ?? f.walkClip;
                            clipSpeed = 3f;
                        }
                        break;
                    case StationKind.Sugar:
                        if (st.cart && st.cartStart && st.cartEnd && st.cartPhase != 0)
                        {
                            // MN9 판정 속도로 설탕을 지고 수레를 끌고 걷는다(돌아올 땐 빈 몸)
                            Vector3 along = st.cartEnd.position - st.cartStart.position;
                            along.y = 0;
                            Vector3 dir = along.normalized;
                            bool going = st.cartPhase == 1;
                            pos = st.cart.position + (going ? dir : -dir) * 36f;
                            pos.y = f.home.y;
                            rot = FaceRotation(f, pos, pos + (going ? dir : -dir));
                            clip = f.walkClip;
                            clipSpeed = going ? Mathf.Clamp(f.walkSpeed * 1.6f * f.BodySpeed, 0.4f, 4f) : 1.8f * f.BodySpeed;
                            carrying = going;
                        }
                        break;
                }
            }
            f.go.transform.position = Vector3.Lerp(f.go.transform.position, pos, Mathf.Clamp01(dt * 10f));
            f.go.transform.rotation = Quaternion.Slerp(f.go.transform.rotation, rot, Mathf.Clamp01(dt * 8f));
            PlayClip(f, clip, clipSpeed);
            if (f.carry)
            {
                f.carry.gameObject.SetActive(carrying);
                if (carrying) f.carry.position = f.go.transform.position + Vector3.up * 12f;
            }
        }
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
            foreach (var f in flies.Where(x => !x.aptDone)) SendAptitude(f);
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
        if (w.st == null)
        {
            OnSoldierBrain(w.fly, type, m);
            return;
        }
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
        float now = Time.time;
        switch (st.kind)
        {
            case StationKind.Sorter:
                st.push = MiniJson.Text(verdict, "action") == "push";
                bool right = verdict != null && verdict.TryGetValue("correct", out var c) && c is bool cb && cb;
                if (right) { st.correct++; Earn(st, SortRight); } else { st.wrong++; Earn(st, SortWrong); }
                if (st.push)
                {
                    fly.pushDur = 0.9f / fly.BodySpeed;
                    st.leverUntil = now + fly.pushDur;
                    fly.pushUntil = now + fly.pushDur;
                    Debug.Log($"BODY 레버 밀기 {fly.name} 다가가기 {MiniJson.Num(fly.values, "approach_hz"):0.#}Hz");
                    EventShot("push", st, 0.45f);
                }
                st.decided = true;
                if (st.boxPhase == 2) { st.boxPhase = 3; st.boxT = 0; st.boxFrom = st.boxObj.position; }
                break;
            case StationKind.Guard:
                bool alarm = MiniJson.Text(verdict, "action") == "alarm";
                string ev = MiniJson.Text(p, "event");
                if (alarm)
                {
                    st.lampUntil = now + 1.5f;
                    st.gateUntil = now + 2.5f;
                    fly.hopStart = now;
                    Debug.Log($"BODY 도약 {fly.name} 거대섬유 {MiniJson.Num(fly.values, "GF_peak50ms_hz"):0.#}Hz 사건 {ev}");
                    EventShot("hop", st, 0.4f);
                    if (ev == "intruder" && intruderActive && !intruderBlocked)
                    {
                        intruderBlocked = true;
                        st.blocked++;
                        eventText = $"{fly.name}(경비 초소 {st.number})의 거대섬유가 켜져 펄쩍 뛰고 침입자를 막았어요";
                    }
                    else if (ev != "intruder")
                    {
                        st.falseAlarm++;
                        Earn(st, FalseAlarmLoss);
                    }
                }
                fly.nextAt = now + Cycle / fly.BodySpeed;   // 장비가 좋으면 하늘을 더 자주 확인한다
                break;
            case StationKind.Sugar:
                double speed = MiniJson.Num(verdict, "speed");
                fly.walkSpeed = (float)speed;
                if (speed < 0.02)
                {
                    st.noReaction++;
                    fly.nextAt = now + Cycle;
                }
                else
                {
                    st.cartDur = 6f / ((float)speed * fly.BodySpeed);
                    Debug.Log($"BODY 수레 끌기 {fly.name} MN9 {MiniJson.Num(fly.values, "MN9_mean_hz"):0.#}Hz → 걷기 {fly.walkSpeed:0.00}배 · 배달 {st.cartDur:0.0}초");
                    st.cartPhase = 1;
                    st.cartT = 0;
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
        HandleShots(now);
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
            if (gameOver || now < st.blockedUntil) continue;   // 괴물이 작업대를 부수는 중이거나 게임 오버
            switch (st.kind)
            {
                case StationKind.Sorter: TickSorter(st, now, dt); break;
                case StationKind.Guard: TickGuard(st, now); break;
                case StationKind.Sugar: TickSugar(st, now, dt); break;
            }
        }
        UpdateFlies(now, dt);
        UpdateDefense(now, dt);
        UpdateCamera(dt);
    }

    void HandleShots(float now)
    {
        if (shotAt > 0 && !shotTaken && now >= shotAt && !string.IsNullOrEmpty(shotPath))
        {
            ScreenCapture.CaptureScreenshot(shotPath);
            shotTaken = true;
            Invoke(nameof(QuitNow), 2f);
        }
        if (shotIndex < shotPlan.Count && !string.IsNullOrEmpty(shotDir))
        {
            var (at, view) = shotPlan[shotIndex];
            if (now >= at - 2.5f)
            {
                ClearDefenseFocus();
                if (view == "fight") focusFight = true;
                else if (view.StartsWith("gate") && int.TryParse(view.Substring(4), out int gi)) focusGate = gi;
                focus = view == "all" || focusFight || focusGate >= 0 ? -1 : int.TryParse(view.TrimStart('s'), out int n) && n < stations.Count ? n : -1;
            }
            if (now >= at)
            {
                ScreenCapture.CaptureScreenshot(Path.Combine(shotDir, $"shot_{shotIndex}_{view}.png"));
                shotIndex++;
                if (shotIndex >= shotPlan.Count) planDoneAt = now;
            }
        }
        // 계획 캡처가 끝나면 몸짓 사건(레버 밀기·도약)을 한 장씩 찍고, 다 찍거나 90초가 지나면 끈다
        if (eventShotAt > 0 && now >= eventShotAt)
        {
            ScreenCapture.CaptureScreenshot(Path.Combine(shotDir, $"event_{eventShotName}.png"));
            eventShotsDone.Add(eventShotName);
            eventShotAt = -1f;
        }
        if (planDoneAt > 0 && eventShotAt < 0 && (eventShotsDone.Count >= ExpectedEventShots || now - planDoneAt > 90f))
        {
            planDoneAt = -1f;
            Invoke(nameof(QuitNow), 2.5f);
        }
    }

    void EventShot(string name, Station st, float delay)
    {
        if (string.IsNullOrEmpty(shotDir) || planDoneAt < 0 || eventShotAt > 0 || eventShotsDone.Contains(name)) return;
        focus = stations.IndexOf(st);
        ClearDefenseFocus();
        camInit = false;   // 카메라를 그 작업대로 바로 옮긴다
        eventShotName = name;
        eventShotAt = Time.time + delay;
    }

    void UpdateCamera(float dt)
    {
        for (int k = 0; k <= 9; k++)
            if (Input.GetKeyDown(KeyCode.Alpha0 + k))
            {
                focus = k == 0 ? -1 : Mathf.Min(k - 1, stations.Count - 1);
                ClearDefenseFocus();
            }
        if (Input.GetMouseButton(1))
        {
            camYaw += Input.GetAxis("Mouse X") * 4f;
            camPitch = Mathf.Clamp(camPitch - Input.GetAxis("Mouse Y") * 3f, 8f, 80f);
        }
        float wheel = Input.GetAxis("Mouse ScrollWheel");
        Vector3 target;
        float wantDist;
        if (DefenseCameraTarget(out Vector3 defTarget, out float defDist))
        {
            target = defTarget;
            wantDist = defDist;
        }
        else if (focus < 0 || focus >= stations.Count)
        {
            target = new Vector3(0f, 10f, 0f);
            wantDist = (roomHalf + YardWidth * 0.45f) * 2.3f;
        }
        else
        {
            var s = stations[focus];
            target = BoundsOf(s.root).center + (s.flySpot ? (s.flySpot.position - s.root.transform.position) * 0.35f : Vector3.zero);
            wantDist = 230f;
        }
        wantDist *= Mathf.Exp(-wheel * 2f);
        float wantYaw = !float.IsNaN(defenseYaw) ? defenseYaw : camYaw + (focus < 0 ? -48f : 0f);
        if (!camInit)
        {
            camTarget = target;
            camDist = wantDist;
            yawNow = wantYaw;
            camInit = true;
        }
        camTarget = Vector3.Lerp(camTarget, target, Mathf.Clamp01(dt * 4f));
        camDist = Mathf.Lerp(camDist, wantDist, Mathf.Clamp01(dt * 4f));
        // 초파리가 서는 쪽(+X, 유니티에서 좌우가 뒤집혀 들어옴) 뒤에서 설비를 내려다본다
        yawNow = Mathf.LerpAngle(yawNow, wantYaw, Mathf.Clamp01(dt * 4f));
        // 오른쪽 작업대 판이 화면 1/3을 가리므로 보는 점을 화면 왼쪽으로 옮긴다
        Vector3 aim = camTarget + Quaternion.Euler(0f, yawNow, 0f) * Vector3.right * camDist * (focus < 0 ? 0.16f : 0.12f);
        cam.transform.position = aim + Quaternion.Euler(!float.IsNaN(defenseYaw) ? camPitch + 23f : focus < 0 ? camPitch + 12f : camPitch, yawNow, 0f) * new Vector3(0, 0, -camDist);
        cam.transform.LookAt(aim);
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
                // 분류대는 적성 검사로 이 초파리의 레버 문턱·연결을 맞춘 뒤부터 일한다(기본 문턱으로 일하면 초반 정확도가 35~47%였다)
                if (st.fly == null || !st.fly.aptDone || st.fly.pending || now < st.fly.nextAt || !brain.Ready) return;
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
                st.boxT += dt * (st.fly != null ? st.fly.BodySpeed : 1f) / 1.0f;
                st.boxObj.position = Vector3.Lerp(st.boxFrom, st.boxTo, st.boxT);
                if (st.boxT >= 1f)
                {
                    st.boxPhase = st.decided ? 3 : 2;
                    st.boxT = 0;
                    st.boxFrom = st.boxObj.position;
                }
                break;
            case 3:
                st.boxT += dt * (st.fly != null ? st.fly.BodySpeed : 1f) / 1.2f;
                Vector3 to = st.push && st.binA ? st.binA.position : st.binB ? st.binB.position : st.boxEnd.position;
                st.boxObj.position = Vector3.Lerp(st.boxFrom, to, Mathf.SmoothStep(0, 1, st.boxT));
                if (st.boxT >= 1f)
                {
                    st.boxObj.gameObject.SetActive(false);
                    st.boxPhase = 0;
                    if (st.fly != null) st.fly.nextAt = now + 0.4f / st.fly.BodySpeed;
                }
                break;
        }
    }

    void TickGuard(Station st, float now)
    {
        if (st.lamp)
        {
            foreach (var r in st.lamp.GetComponentsInChildren<Renderer>())
            {
                bool on = now < st.lampUntil && Mathf.Repeat(now * 6f, 1f) < 0.6f;
                r.material.EnableKeyword("_EMISSION");
                r.material.SetColor("_EmissionColor", on ? new Color(3f, 0.3f, 0.2f) : new Color(0.15f, 0.02f, 0.02f));
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
                st.cart.position = Vector3.Lerp(st.cart.position, st.cartStart.position, dt * 8f);
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
                st.cartT += dt * (st.fly != null ? st.fly.BodySpeed : 1f) / 2.5f;
                st.cart.position = Vector3.Lerp(st.cartEnd.position, st.cartStart.position, st.cartT);
                if (st.cartT >= 1f) { st.cartPhase = 0; if (st.fly != null) st.fly.nextAt = now + 0.3f; }
                break;
        }
    }

    void QuitNow() => Application.Quit();

    // ---------------- 화면 ----------------
    double StationCost => Math.Round(150 * Math.Pow(1.4, stationsBought));   // 작업대를 사면 초파리 한 마리가 함께 온다

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

        GUILayout.BeginArea(new Rect(14, 14, 460, Mathf.Min(780, Screen.height - 28)), GUI.skin.box);
        GUILayout.Label("초파리 공장", title);
        GUILayout.Label($"돈 <b>{money:0}원</b>   최근 1분 {perMin:+0;-0;0}원", body);
        GUILayout.Label(brain.Connected ? $"뇌 서버 연결됨 · 계산 프로세스 {brain.WorkersReady}/{brain.Workers} · 대기열 {brain.Queue}" : "뇌 서버 연결 기다리는 중(factory/server/brain_server.py)", small);
        GUILayout.Label("초파리는 조종도 학습도 되지 않아요. 타고난 반사 뉴런(다가가기·도주·섭식) 발화로만 일해요. 걷기·날갯짓 모양은 Blender 동작이고, 언제·얼마나 빨리 움직일지는 뇌가 정해요.", small);
        GUILayout.Label("장비 강화는 몸 동작(상자 옮기기·수레 끌기·하늘 확인 주기)만 빠르게 해요. 레버를 밀지·뛸지·수레 속도 비율은 그대로 뇌 계산이라, 몸이 빨라지면 뇌 대기열이 병목이 돼요.", small);
        GUILayout.Label($"<i>{eventText}</i>", small);
        GUILayout.Label("카메라: 0 전체 · 1~9 작업대 확대 · 휠 줌 · 오른쪽 드래그 회전", small);
        GUILayout.Space(4);
        GUILayout.Label("작업대를 사면 초파리 한 마리가 함께 와요", small);
        GUILayout.BeginHorizontal();
        foreach (var (kind, label) in new[] { (StationKind.Sorter, "분류대"), (StationKind.Guard, "경비 초소"), (StationKind.Sugar, "설탕대") })
        {
            if (GUILayout.Button($"{label} {StationCost}원") && money >= StationCost)
            {
                money -= StationCost;
                stationsBought++;
                var st = AddStation(kind);
                var idle = flies.FirstOrDefault(f => f.station == null && !f.soldier);
                if (idle != null) Assign(idle, st);
                else AddFly(st);
                PlaceFlies();
            }
        }
        GUILayout.EndHorizontal();
        GUILayout.Label($"설탕 농도(당 감각뉴런 자극) {sugarHz:0}Hz · 한 번에 {sugarHz * SugarCostPerHz:0.0}원", small);
        sugarHz = Mathf.Round(GUILayout.HorizontalSlider(sugarHz, 50f, 200f) / 10f) * 10f;
        var idleFlies = flies.Where(f => f.station == null && !f.soldier).ToList();
        if (idleFlies.Count > 0) GUILayout.Label("쉬는 초파리: " + string.Join(", ", idleFlies.Select(f => f.name)), small);
        DefenseGUI();
        GUILayout.EndArea();

        float w = 520, x = Screen.width - w - 14;
        GUILayout.BeginArea(new Rect(x, 14, w, Screen.height - 28), GUI.skin.box);
        scroll = GUILayout.BeginScrollView(scroll);
        for (int i = 0; i < stations.Count; i++)
        {
            var st = stations[i];
            var f = st.fly;
            GUILayout.Label($"<b>{i + 1}. {st.Name}</b> — {(f != null ? $"{f.name} (개체 {f.seed})" : "초파리 없음")}", body);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("보기", GUILayout.Width(60))) { focus = i; ClearDefenseFocus(); }
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
                GUILayout.BeginHorizontal();
                GUILayout.Label($"장비: {GearName[f.gear]} · 몸 속도 ×{f.BodySpeed:0.00}", small, GUILayout.Width(220));
                if (f.gear < 3)
                {
                    int cost = GearCost[f.gear];
                    if (GUILayout.Button($"강화 → {GearName[f.gear + 1]} {cost}원", GUILayout.Width(220)) && money >= cost)
                    {
                        money -= cost;
                        f.gear++;
                        ApplyGear(f);
                        eventText = $"{f.name} 장비 강화: {GearName[f.gear]} · 몸 속도 ×{f.BodySpeed:0.00}";
                    }
                }
                else GUILayout.Label("최고 단계", small);
                GUILayout.EndHorizontal();
                Rect r = GUILayoutUtility.GetRect(w - 40, 24);
                var v = f.values;
                if (st.kind == StationKind.Sorter) Bar(r, "다가가기", MiniJson.Num(v, "approach_hz"), f.leverThreshold, 40, new Color(0.9f, 0.45f, 0.3f));
                else if (st.kind == StationKind.Guard) Bar(r, "도주 거대섬유", MiniJson.Num(v, "GF_peak50ms_hz"), GfThreshold, 200, new Color(0.35f, 0.6f, 0.95f));
                else Bar(r, "섭식 MN9", MiniJson.Num(v, "MN9_mean_hz"), Mn9Ref, 90, new Color(0.55f, 0.8f, 0.45f));
                string waitNote = st.kind == StationKind.Sorter && !f.aptDone ? "적성 검사가 끝나면 분류를 시작해요 · " : "";
                GUILayout.Label(waitNote + (f.pending ? "뇌 계산 중… " : "왜: ") + f.reason + (f.outcome != "" ? $" → {f.outcome}" : ""), small);
                GUILayout.Label(f.aptText, small);
            }
            GUILayout.Space(8);
        }
        SoldierRowsGUI(w);
        GUILayout.EndScrollView();
        GUILayout.EndArea();
        DefenseWorldGUI();
        GameOverGUI();
    }

    void Swap(Station st)
    {
        var idle = flies.FirstOrDefault(f => f.station == null && !f.soldier);
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
