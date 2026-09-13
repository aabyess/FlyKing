using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// 공장 방어 — 사방 문을 부수러 오는 괴물 · 총 쏘는 병정 초파리 · 공장이 무너지면 게임 오버.
///
/// 병정도 조종·학습이 없다. 가장 가까운 괴물이 병정 눈에 보이는 장면(factory/brain/stimuli.soldier)을 뇌에 넣고 반사 뉴런으로만 움직인다.
///   멀리 있는 괴물 = 작게 움직이는 것(small) · 가까이 덮치는 거미 = 루밍(loom)
///   거대섬유 ≥ 60Hz → 도약 도주(사격 멈춤) / 다가가기 ≥ 5Hz → 방아쇠 / 둘 다 아님 → 제자리
///   조준 = 방향 틀기(DNa02 왼쪽−오른쪽) 부호. 괴물 쪽과 반대면 거울 쪽으로 쏴서 빗나간다.
/// 실측(factory/brain/soldier_probe.json, 씨앗 5): 좀비 개미 다가가기 왼쪽 21.5·오른쪽 9.9 vs 빈 바닥 0.1Hz,
///   거미 루밍 거대섬유 10/10 ≥ 60Hz · 개미 0/10, 방향 틀기 부호가 괴물 쪽과 맞음 개미 10/10 · 거미 9/10.
/// 총은 게임 소품이다(초파리는 총을 모른다) — 방아쇠 시점과 조준 쪽만 뇌가 정한다.
/// 모델: 괴물 factory/blender/gen_monsters.py · 병정 총·철모 gen_armor.py · 문 gen_room.py.
/// </summary>
public partial class Fly
{
    public bool soldier;
    public int post, slot;
    public readonly List<GameObject> soldierGear = new List<GameObject>();
    public Transform muzzle;
    public Monster target;
    public string soldierAction = "대기", sideAsked = "", viewAsked = "";
    public float actionUntil, nextShotAt;
    public int shotsLeft;
    public Vector3 aimDir, moveDir, logicPos;
    public int kills, bursts, shots, hits, flees, holds;
}

public partial class Station
{
    public float blockedUntil;
}

public class Monster
{
    public string kind;                    // ant · spider
    public GameObject go;
    public readonly List<(Transform t, Quaternion rest, float phase)> legs = new List<(Transform, Quaternion, float)>();
    public Vector3 forwardModel = Vector3.right, baseScale = Vector3.one;
    public float hp, maxHp, speed, legClock, attackAt, dieAt = -1f, hitFlash;
    public int gate;
    public bool inside;
    public Station prey;
    public string Name => kind == "ant" ? "좀비 개미" : "거미 괴물";
}

public class Gate
{
    public int index;
    public float hp = 60f, maxHp = 60f, shakeUntil;
    public GameObject door;
    public Quaternion doorRest;
    public Vector3 doorPos;
    public bool Broken => hp <= 0f;
}

public class Bullet
{
    public Vector3 pos, dir;
    public float life;
    public Fly owner;
    public GameObject go;
}

public partial class FactoryGame
{
    const float GateHalf = 100f, YardWidth = 700f, SightRange = 650f, LoomRange = 230f, BulletSpeed = 900f, SoldierFireHz = 5f;
    const double FactoryHpMax = 300;
    static readonly string[] GateName = { "서쪽", "동쪽", "남쪽", "북쪽" };
    static Material bulletMat;

    readonly List<Monster> monsters = new List<Monster>();
    readonly List<Gate> gates = new List<Gate>();
    readonly List<Bullet> bullets = new List<Bullet>();
    bool testSoldiers, focusFight, gameOver;
    int focusGate = -1;
    float defenseYaw = float.NaN;   // 싸움·문 보기일 때 카메라 방향(병정 뒤에서 문 밖을 보게), 아니면 NaN
    double factoryHp = FactoryHpMax;
    Fly fightFly;
    int wave, soldiersBought, totalKills;
    float nextWaveAt, muzzleUntil;
    int[] nextWaveGates = { 0 };
    string defenseText = "";
    Light muzzleLight;

    double SoldierCost => Math.Round(120 * Math.Pow(1.3, soldiersBought));
    int ExpectedEventShots => testSoldiers ? 6 : 2;

    static FactoryGame()
    {
        // factory/blender/gen_monsters.py
        MatSpec["몬스터_좀비피부"] = (null, new Color(0.22f, 0.28f, 0.19f), 0f, 0.85f);
        MatSpec["몬스터_썩은살"] = (null, new Color(0.36f, 0.15f, 0.16f), 0f, 0.6f);
        MatSpec["몬스터_좀비다리"] = (null, new Color(0.12f, 0.14f, 0.10f), 0f, 0.8f);
        MatSpec["몬스터_초록눈"] = (null, new Color(0.45f, 1.0f, 0.25f), 0f, 0.3f);
        MatSpec["몬스터_거미털"] = (null, new Color(0.06f, 0.055f, 0.055f), 0f, 0.95f);
        MatSpec["몬스터_거미무늬"] = (null, new Color(0.55f, 0.03f, 0.02f), 0f, 0.5f);
        MatSpec["몬스터_빨간눈"] = (null, new Color(1.0f, 0.1f, 0.05f), 0f, 0.2f);
        MatSpec["몬스터_송곳니"] = (null, new Color(0.75f, 0.72f, 0.62f), 0f, 0.35f);
        // factory/blender/gen_armor.py — 병정 장비
        MatSpec["병정_총몸"] = (null, new Color(0.12f, 0.12f, 0.13f), 0.8f, 0.4f);
        MatSpec["병정_총열"] = (null, new Color(0.25f, 0.25f, 0.27f), 1f, 0.3f);
        MatSpec["병정_개머리"] = (null, new Color(0.32f, 0.20f, 0.10f), 0f, 0.7f);
        MatSpec["병정_조준경"] = (null, new Color(0.04f, 0.04f, 0.045f), 0.6f, 0.3f);
        MatSpec["병정_철모"] = (null, new Color(0.24f, 0.28f, 0.16f), 0f, 0.8f);
    }

    static Vector3 Flat(Vector3 v)
    {
        v.y = 0f;
        return v;
    }

    static Quaternion Yaw(Vector3 from, Vector3 to) => Quaternion.Euler(0f, Vector3.SignedAngle(Flat(from), Flat(to), Vector3.up), 0f);

    static Vector3 GateDir(int g) => g == 0 ? Vector3.left : g == 1 ? Vector3.right : g == 2 ? Vector3.back : Vector3.forward;
    Vector3 GatePoint(int g) => GateDir(g) * roomHalf;

    static void ApplySoldierGear(Fly f)
    {
        foreach (var g in f.soldierGear) g.SetActive(f.soldier && !(g.name.Contains("철모") && f.gear >= 2));   // 갑옷 투구가 있으면 철모는 벗는다
    }

    void ClearDefenseFocus()
    {
        focusFight = false;
        focusGate = -1;
    }

    // ---------------- 시작·방·문 ----------------
    void StartDefense()
    {
        muzzleLight = new GameObject("총구 불빛").AddComponent<Light>();
        muzzleLight.type = LightType.Point;
        muzzleLight.range = 90f;
        muzzleLight.intensity = 0f;
        muzzleLight.color = new Color(1f, 0.8f, 0.4f);
        if (testSoldiers)
        {
            // 검증 캡처용: 서·동·남 문에 병정(쇠 판금), 습격은 서쪽(지키는 문)과 북쪽(빈 문)
            // -testFactoryHp로 내구도를 낮춰 시작하면 게임 오버 화면 확인용이라 병정을 두지 않는다(병정이 들어온 괴물을 다 잡으면 게임 오버에 못 닿는다)
            foreach (int g in factoryHp < FactoryHpMax ? new int[0] : new[] { 0, 1, 2 })
            {
                var s = AddSoldier(g);
                s.gear = 2;
                ApplyGear(s);
            }
            nextWaveGates = new[] { 0, 3 };
            nextWaveAt = Time.time + 10f;
        }
        else
        {
            nextWaveAt = Time.time + 90f;
            PlanNextWave();
        }
    }

    void BuildYardAndGates(Vector3 tileScale)
    {
        // 방 둘레 마당(괴물이 걸어오는 길): 같은 콘크리트 타일을 두 배 크기로, 방 바닥보다 조금 낮게(겹쳐도 깜빡이지 않게)
        var tile = Resources.Load<GameObject>("Models/floor_concrete") ?? Resources.Load<GameObject>("Models/floor_tile");
        var lamp = Resources.Load<GameObject>("Models/lamp_hanging");
        var doorPrefab = Resources.Load<GameObject>("Models/gate_steel");
        float outer = roomHalf + YardWidth;
        if (tile)
        {
            int n = Mathf.CeilToInt(outer * 2f / 200f);
            for (int ix = 0; ix < n; ix++)
                for (int iz = 0; iz < n; iz++)
                {
                    float x = -outer + ix * 200f + 100f, z = -outer + iz * 200f + 100f;
                    if (Mathf.Abs(x) + 100f <= roomHalf && Mathf.Abs(z) + 100f <= roomHalf) continue;
                    Instantiate(tile, new Vector3(x, -0.4f, z), Quaternion.identity, worldRoot).transform.localScale = tileScale * 2f;
                }
        }
        while (gates.Count < 4) gates.Add(new Gate { index = gates.Count });
        foreach (var gt in gates)
        {
            bool near = gt.index == 1 || gt.index == 3;                 // 카메라 쪽 문은 벽처럼 낮게
            gt.doorRest = Quaternion.Euler(0f, gt.index <= 1 ? 90f : 0f, 0f);
            gt.doorPos = GatePoint(gt.index);
            if (doorPrefab)
            {
                gt.door = Instantiate(doorPrefab, gt.doorPos, gt.doorRest, worldRoot);
                FitWidth(gt.door, 190f, "문");
                if (near)
                {
                    var s = gt.door.transform.localScale;
                    gt.door.transform.localScale = new Vector3(s.x, s.y * 0.35f, s.z);
                }
            }
            else
            {
                gt.door = GameObject.CreatePrimitive(PrimitiveType.Cube);
                gt.door.transform.SetParent(worldRoot, false);
                gt.doorPos += Vector3.up * (near ? 12f : 35f);
                gt.door.transform.SetPositionAndRotation(gt.doorPos, gt.doorRest);
                gt.door.transform.localScale = new Vector3(190f, near ? 24f : 70f, 6f);
            }
            Vector3 top = GatePoint(gt.index) + GateDir(gt.index) * 130f + Vector3.up * 150f;
            if (lamp) FitWidth(Instantiate(lamp, top, Quaternion.identity, worldRoot), 32f, "문 전등");
            var spot = new GameObject("문 전등빛").AddComponent<Light>();
            spot.transform.SetParent(worldRoot, false);
            spot.transform.position = top + Vector3.up * 6f;
            spot.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            spot.type = LightType.Spot;
            spot.range = 480f;
            spot.spotAngle = 100f;
            spot.intensity = 2.0f;
            spot.color = new Color(0.8f, 0.88f, 1f);
        }
    }

    void UpdateGates(float now, float dt)
    {
        foreach (var gt in gates)
        {
            if (gt.door == null) continue;
            var t = gt.door.transform;
            if (gt.Broken)
            {
                float sign = gt.index == 1 || gt.index == 3 ? 1f : -1f;       // 바깥쪽으로 쓰러진다
                t.rotation = Quaternion.Slerp(t.rotation, gt.doorRest * Quaternion.Euler(80f * sign, 0f, 0f), Mathf.Clamp01(dt * 5f));
            }
            else
            {
                Vector3 shake = now < gt.shakeUntil ? GateDir(gt.index) * Mathf.Sin(now * 60f) * 2.5f : Vector3.zero;
                t.SetPositionAndRotation(gt.doorPos + shake, Quaternion.Slerp(t.rotation, gt.doorRest, Mathf.Clamp01(dt * 5f)));
            }
        }
    }

    // ---------------- 습격 ----------------
    void PlanNextWave()
    {
        int dirs = Mathf.Min(4, 1 + wave / 3);
        nextWaveGates = Enumerable.Range(0, 4).OrderBy(_ => env.Next()).Take(dirs).ToArray();
    }

    void SpawnWave()
    {
        wave++;
        foreach (int g in nextWaveGates)
        {
            int ants = 2 + wave / 2;
            for (int i = 0; i < ants; i++) SpawnMonster("ant", g, i * 70f);
            if (wave % 3 == 0 || testSoldiers) SpawnMonster("spider", g, ants * 70f + 120f);
        }
        defenseText = $"습격 {wave}: {string.Join("·", nextWaveGates.Select(i => GateName[i]))} 문으로 괴물이 몰려와요!";
        Debug.Log($"DEFENSE 습격 {wave} 문 {string.Join(",", nextWaveGates)}");
        nextWaveAt = Time.time + (testSoldiers ? 60f : 45f);
        PlanNextWave();
    }

    void SpawnMonster(string kind, int gate, float behind)
    {
        var m = new Monster { kind = kind, gate = gate };
        var prefab = Resources.Load<GameObject>(kind == "ant" ? "Models/monster_zombie_ant" : "Models/monster_spider");
        if (prefab != null)
        {
            m.go = Instantiate(prefab);
            FitWidth(m.go, kind == "ant" ? 62f : 170f, m.Name);
            DressMaterials(m.go);
        }
        else
        {
            m.go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            m.go.transform.localScale = Vector3.one * (kind == "ant" ? 30f : 80f);
        }
        m.baseScale = m.go.transform.localScale;
        Transform head = null;
        foreach (var t in m.go.GetComponentsInChildren<Transform>())
        {
            if (t.name.Contains("_다리_")) m.legs.Add((t, t.localRotation, (m.legs.Count % 2) * Mathf.PI + m.legs.Count * 0.3f));
            if (t.name.EndsWith("_머리")) head = t;
        }
        if (head)
        {
            Vector3 fwd = Flat(head.position - m.go.transform.position);
            if (fwd.sqrMagnitude > 1e-6f) m.forwardModel = fwd.normalized;
        }
        Vector3 dir = GateDir(gate), side = Vector3.Cross(Vector3.up, dir);
        float lateral = (float)(env.NextDouble() * 2 - 1) * 180f;
        m.go.transform.position = dir * (roomHalf + YardWidth + behind) + side * lateral;
        m.go.transform.rotation = Yaw(m.forwardModel, -dir);
        m.maxHp = m.hp = kind == "ant" ? 3f : 14f;
        m.speed = kind == "ant" ? 42f : 30f;
        monsters.Add(m);
    }

    void FaceMonster(Monster m, Vector3 dir, float dt)
    {
        if (dir.sqrMagnitude < 1e-4f) return;
        m.go.transform.rotation = Quaternion.Slerp(m.go.transform.rotation, Yaw(m.forwardModel, dir), Mathf.Clamp01(dt * 6f));
    }

    void UpdateMonsters(float now, float dt)
    {
        for (int i = monsters.Count - 1; i >= 0; i--)
        {
            var m = monsters[i];
            var tr = m.go.transform;
            if (m.dieAt > 0f)
            {
                float k = Mathf.Clamp01((now - m.dieAt) / 1.2f);
                tr.localScale = m.baseScale * (1f - k * 0.9f);
                tr.position += Vector3.down * 20f * dt;
                if (k >= 1f)
                {
                    Destroy(m.go);
                    monsters.RemoveAt(i);
                }
                continue;
            }
            var gt = gates[m.gate];
            Vector3 pos = tr.position;
            float reach = m.kind == "ant" ? 40f : 95f;
            Vector3 goal;
            if (!m.inside)
            {
                goal = gt.Broken ? GatePoint(m.gate) - GateDir(m.gate) * 30f : GatePoint(m.gate) + GateDir(m.gate) * (reach * 0.6f);
                if (gt.Broken && Flat(pos - GatePoint(m.gate)).magnitude < 45f) m.inside = true;
            }
            else
            {
                if (m.prey == null) m.prey = stations.OrderBy(s => Flat(s.root.transform.position - pos).sqrMagnitude).FirstOrDefault();
                goal = m.prey != null ? BoundsOf(m.prey.root).center : Vector3.zero;
            }
            Vector3 to = Flat(goal - pos);
            bool working = false;
            if (!m.inside && !gt.Broken && to.magnitude < 12f)
            {
                working = true;                                         // 문 두드리기
                FaceMonster(m, -GateDir(m.gate), dt);
                if (now >= m.attackAt)
                {
                    m.attackAt = now + (m.kind == "ant" ? 1.2f : 0.9f);
                    gt.hp = Mathf.Max(0f, gt.hp - (m.kind == "ant" ? 2f : 9f));
                    gt.shakeUntil = now + 0.25f;
                    if (gt.Broken)
                    {
                        defenseText = $"{GateName[m.gate]} 문이 부서졌어요! 괴물이 공장 안으로 들어와요";
                        Debug.Log($"DEFENSE 문 부서짐 {GateName[m.gate]}");
                        DefenseShot("gate", 0.8f, null, m.gate);
                    }
                }
            }
            else if (m.inside && m.prey != null && to.magnitude < reach + 40f)
            {
                working = true;                                         // 작업대 부수기
                FaceMonster(m, to, dt);
                m.prey.blockedUntil = now + 0.6f;
                if (now >= m.attackAt)
                {
                    m.attackAt = now + (m.kind == "ant" ? 1.5f : 1.0f);
                    double dmg = m.kind == "ant" ? 4 : 15;
                    factoryHp = Math.Max(0, factoryHp - dmg);
                    defenseText = $"{m.Name}가 {m.prey.Name}를 부숴요(공장 내구도 −{dmg})";
                    if (factoryHp <= 0) GameOver();
                }
            }
            else if (to.sqrMagnitude > 1f)
            {
                tr.position = pos + to.normalized * Mathf.Min(m.speed * dt, to.magnitude);
                FaceMonster(m, to, dt);
            }
            m.legClock += dt * (working ? 5f : 9f);
            float swing = working ? 10f : 24f;
            foreach (var (t, rest, phase) in m.legs)
                t.localRotation = rest * Quaternion.Euler(0f, Mathf.Sin(m.legClock + phase) * swing, 0f);
            if (m.hitFlash > 0f) m.hitFlash -= dt;
            tr.localScale = m.baseScale * (1f + Mathf.Max(0f, m.hitFlash) * 0.8f);
        }
    }

    void GameOver()
    {
        if (gameOver) return;
        gameOver = true;
        defenseText = "공장이 무너졌어요 — 게임 오버";
        Debug.Log($"DEFENSE 게임 오버 습격 {wave} 처치 {totalKills}");
        DefenseShot("gameover", 0.6f);
    }

    // ---------------- 병정 ----------------
    int NextPost()
    {
        int[] c = new int[4];
        foreach (var s in flies.Where(x => x.soldier)) c[s.post]++;
        return Array.IndexOf(c, c.Min());
    }

    Fly AddSoldier(int post = -1)
    {
        if (post < 0) post = NextPost();
        int slot = flies.Count(x => x.soldier && x.post == post);
        var f = AddFly(null, true);
        f.post = post;
        f.slot = slot;
        f.logicPos = PostPoint(post, slot);
        f.go.transform.SetPositionAndRotation(f.logicPos, Yaw(f.forwardModel, GateDir(post)));
        f.muzzle = FindDeep(f.go.transform, "병정_총구");
        return f;
    }

    Vector3 PostPoint(int g, int slot)
    {
        Vector3 d = GateDir(g), side = Vector3.Cross(Vector3.up, d);
        int col = slot % 3, row = slot / 3;
        return GatePoint(g) - d * (90f + row * 60f) + side * ((col - 1) * 60f);
    }

    void UpdateDefense(float now, float dt)
    {
        if (!gameOver && now >= nextWaveAt) SpawnWave();
        UpdateGates(now, dt);
        if (!gameOver) UpdateMonsters(now, dt);
        UpdateBullets(now, dt);
        foreach (var f in flies)
            if (f.soldier) UpdateSoldier(f, now, dt);
    }

    void UpdateSoldier(Fly f, float now, float dt)
    {
        var tr = f.go.transform;
        Vector3 home = PostPoint(f.post, f.slot);
        Vector3 pos = f.logicPos, offset = Vector3.zero;
        Quaternion rot = Yaw(f.forwardModel, GateDir(f.post));
        string clip = f.idleClip;
        float clipSpeed = 1f;
        bool acting = now < f.actionUntil;
        if (f.pending && now - f.pendingSince > 30f) f.pending = false;
        if (acting && f.soldierAction == "도망")
        {
            float k = 1f - (f.actionUntil - now) / 1.2f;
            pos += f.moveDir * 140f * f.BodySpeed * dt;
            offset.y = Mathf.Sin(Mathf.PI * Mathf.Clamp01(k * 1.4f)) * 45f;   // 거대섬유 도약
            rot = Yaw(f.forwardModel, -f.moveDir);
            clip = f.flyClip ?? f.walkClip;
            clipSpeed = 3f;
        }
        else if (f.soldierAction == "사격" && (acting || f.shotsLeft > 0))
        {
            rot = Yaw(f.forwardModel, f.aimDir);
            if (f.shotsLeft > 0 && now >= f.nextShotAt)
            {
                FireBullet(f, now);
                f.shotsLeft--;
                f.nextShotAt = now + 0.16f / f.BodySpeed;
                offset -= f.aimDir * 3f;                                         // 반동
            }
        }
        else
        {
            if (f.soldierAction != "대기" && !acting) f.soldierAction = "대기";
            Vector3 back = Flat(home - pos);
            if (back.magnitude > 6f)
            {
                pos += back.normalized * Mathf.Min(80f * f.BodySpeed * dt, back.magnitude);
                rot = Yaw(f.forwardModel, back);
                clip = f.walkClip;
                clipSpeed = 1.8f * f.BodySpeed;
            }
        }
        f.logicPos = pos;
        tr.position = Vector3.Lerp(tr.position, pos + offset, Mathf.Clamp01(dt * 14f));
        tr.rotation = Quaternion.Slerp(tr.rotation, rot, Mathf.Clamp01(dt * 10f));
        PlayClip(f, clip, clipSpeed);

        if (gameOver || f.pending || now < f.nextAt || !brain.Ready || acting || f.shotsLeft > 0) return;
        Monster target = null;
        float best = SightRange;
        foreach (var x in monsters)
        {
            if (x.dieAt > 0f || (x.gate != f.post && !x.inside)) continue;   // 벽 너머 다른 문 괴물은 안 보인다
            float d = Flat(x.go.transform.position - pos).magnitude;
            if (d < best)
            {
                best = d;
                target = x;
            }
        }
        if (target != null) RequestSoldier(f, target, pos, best);
    }

    void RequestSoldier(Fly f, Monster m, Vector3 pos, float dist)
    {
        Vector3 to = Flat(m.go.transform.position - pos);
        Vector3 facing = Flat(f.go.transform.rotation * f.forwardModel);
        Vector3 right = Vector3.Cross(Vector3.up, facing);
        string side = Vector3.Dot(to, right) >= 0f ? "right" : "left";
        string view = m.kind == "spider" && dist < LoomRange ? "loom" : "small";   // 멀리 있으면 거미도 작게 움직이는 점으로 보인다
        string id = $"d{++requestN}";
        waiting[id] = (f, null);
        f.pending = true;
        f.pendingSince = Time.time;
        f.target = m;
        f.sideAsked = side;
        f.viewAsked = view;
        brain.Send(new Dictionary<string, object>
        {
            ["type"] = "decide", ["id"] = id, ["fly_seed"] = (double)f.seed, ["station"] = "soldier",
            ["params"] = new Dictionary<string, object> { ["monster"] = m.kind, ["view"] = view, ["side"] = side, ["seed"] = (double)(requestN * 7 + f.seed) },
        });
    }

    void OnSoldierBrain(Fly f, string type, Dictionary<string, object> msg)
    {
        f.pending = false;
        f.decisions++;
        float now = Time.time;
        if (type == "error")
        {
            f.reason = "뇌 계산 오류: " + MiniJson.Text(msg, "error");
            f.nextAt = now + 1.5f;
            return;
        }
        f.values = MiniJson.Obj(msg, "values");
        f.reason = MiniJson.Text(msg, "reason");
        var verdict = MiniJson.Obj(msg, "verdict");
        f.outcome = MiniJson.Text(verdict, "outcome");
        string action = MiniJson.Text(verdict, "action");
        string turn = MiniJson.Text(verdict, "turn");
        var m = f.target;
        if (gameOver || m == null || m.dieAt > 0f)
        {
            f.nextAt = now + 0.2f;
            return;
        }
        Vector3 to = Flat(m.go.transform.position - f.logicPos).normalized;
        Vector3 facing = Flat(f.go.transform.rotation * f.forwardModel).normalized;
        switch (action)
        {
            case "flee":
                f.soldierAction = "도망";
                f.flees++;
                f.actionUntil = now + 1.2f;
                f.shotsLeft = 0;
                f.moveDir = -to;
                Debug.Log($"BODY 병정 도망 {f.name} 거대섬유 {MiniJson.Num(f.values, "GF_peak50ms_hz"):0.#}Hz {m.Name}({f.viewAsked})");
                DefenseShot("flee", 0.35f, f);
                break;
            case "fire":
            {
                // 조준 = 뇌의 방향 틀기 부호. 괴물 쪽과 같으면 괴물을 겨누고, 반대면 거울 쪽(없으면 정면)으로 쏴서 빗나간다.
                bool agrees = turn == f.sideAsked;
                Vector3 aim = to;
                if (!agrees)
                    aim = turn == "none" ? facing : Quaternion.Euler(0f, -Vector3.SignedAngle(facing, to, Vector3.up), 0f) * facing;
                f.aimDir = aim.normalized;
                f.soldierAction = "사격";
                f.bursts++;
                f.shotsLeft = 3 + f.gear;
                f.nextShotAt = now + 0.15f;                                      // 몸을 돌릴 틈
                f.actionUntil = now + 0.15f;
                Debug.Log($"BODY 병정 사격 {f.name} 다가가기 {MiniJson.Num(f.values, "approach_hz"):0.#}Hz 방향 {turn}(괴물 {f.sideAsked}) {(agrees ? "조준 맞음" : "빗나감")}");
                DefenseShot("fire", 0.45f, f);
                break;
            }
            default:
                f.soldierAction = "제자리";
                f.holds++;
                f.actionUntil = now + 0.6f;
                f.nextAt = now + 0.8f;
                break;
        }
    }

    void FireBullet(Fly f, float now)
    {
        Vector3 origin = f.muzzle != null && f.muzzle.gameObject.activeInHierarchy
            ? f.muzzle.position
            : f.go.transform.position + Vector3.up * 14f + f.aimDir * 26f;
        Vector3 dir = Quaternion.Euler(0f, (float)(env.NextDouble() * 2 - 1) * 2.5f, 0f) * f.aimDir;
        var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        Destroy(go.GetComponent<Collider>());
        go.transform.localScale = new Vector3(1.6f, 6f, 1.6f);
        go.transform.SetPositionAndRotation(origin, Quaternion.FromToRotation(Vector3.up, dir));
        if (bulletMat == null)
        {
            bulletMat = new Material(go.GetComponent<Renderer>().sharedMaterial) { color = new Color(1f, 0.85f, 0.3f) };
            bulletMat.EnableKeyword("_EMISSION");
            bulletMat.SetColor("_EmissionColor", new Color(1f, 0.7f, 0.2f) * 3f);
        }
        go.GetComponent<Renderer>().sharedMaterial = bulletMat;
        bullets.Add(new Bullet { pos = origin, dir = dir, life = 0.9f, owner = f, go = go });
        f.shots++;
        if (muzzleLight != null)
        {
            muzzleLight.transform.position = origin;
            muzzleLight.intensity = 4f;
            muzzleUntil = now + 0.05f;
        }
    }

    static float DistToSegment(Vector3 p, Vector3 a, Vector3 b)
    {
        Vector3 ab = b - a;
        float t = Mathf.Clamp01(Vector3.Dot(p - a, ab) / Mathf.Max(ab.sqrMagnitude, 1e-6f));
        return Vector3.Distance(p, a + ab * t);
    }

    void UpdateBullets(float now, float dt)
    {
        if (muzzleLight != null && now >= muzzleUntil) muzzleLight.intensity = 0f;
        for (int i = bullets.Count - 1; i >= 0; i--)
        {
            var b = bullets[i];
            Vector3 next = b.pos + b.dir * BulletSpeed * dt;
            Monster hit = null;
            foreach (var m in monsters)
            {
                if (m.dieAt > 0f) continue;
                Vector3 c = m.go.transform.position + Vector3.up * (m.kind == "ant" ? 10f : 25f);
                if (DistToSegment(c, b.pos, next) < (m.kind == "ant" ? 24f : 60f))
                {
                    hit = m;
                    break;
                }
            }
            b.pos = next;
            b.life -= dt;
            b.go.transform.position = b.pos;
            if (hit != null)
            {
                b.owner.hits++;
                hit.hp -= 1f;
                hit.hitFlash = 0.12f;
                if (hit.hp <= 0f) Kill(b.owner, hit, now);
            }
            if (hit != null || b.life <= 0f)
            {
                Destroy(b.go);
                bullets.RemoveAt(i);
            }
        }
    }

    void Kill(Fly f, Monster m, float now)
    {
        m.dieAt = now;
        f.kills++;
        totalKills++;
        double bounty = m.kind == "ant" ? 5 : 35;
        Earn(null, bounty);
        defenseText = $"병정 {f.name}: {m.Name} 처치(+{bounty}원)";
        Debug.Log($"BODY 병정 처치 {f.name} {m.Name}");
        DefenseShot("kill", 0.1f, f);
    }

    // ---------------- 카메라·캡처 ----------------
    void DefenseShot(string name, float delay, Fly f = null, int gate = -1)
    {
        if (string.IsNullOrEmpty(shotDir) || planDoneAt < 0 || eventShotAt > 0 || eventShotsDone.Contains(name)) return;
        ClearDefenseFocus();
        focusFight = f != null;
        fightFly = f;
        focusGate = gate;
        focus = -1;
        camInit = false;
        eventShotName = name;
        eventShotAt = Time.time + delay;
    }

    // 카메라는 보는 점에서 (−sin yaw, −cos yaw) 쪽에 선다 → 문 바깥 방향의 반대(방 안)에 서도록 고른 yaw.
    // 1차(비스듬한 전체 보기 각도 그대로)는 문 옆 기둥이 가리고 싸움이 화면 밖으로 밀렸다.
    static float GateCamYaw(int g) => g == 0 ? -90f : g == 1 ? 90f : g == 2 ? 180f : 0f;

    bool DefenseCameraTarget(out Vector3 target, out float dist)
    {
        target = Vector3.zero;
        dist = 0f;
        defenseYaw = float.NaN;
        if (focusGate >= 0 && focusGate < 4)
        {
            target = GatePoint(focusGate) + GateDir(focusGate) * 50f;
            dist = 460f;
            defenseYaw = GateCamYaw(focusGate);
            return true;
        }
        if (!focusFight) return false;
        var f = fightFly;
        bool stale = f == null || !f.soldier || f.target == null || (f.target.dieAt > 0f && Time.time - f.target.dieAt > 2f);
        if (stale)
        {
            f = flies.Where(x => x.soldier && x.target != null && x.target.dieAt < 0f && x.target.go != null)
                     .OrderBy(x => Flat(x.target.go.transform.position - x.logicPos).sqrMagnitude).FirstOrDefault();
            if (f != null) fightFly = f;
        }
        if (f != null && f.target != null && f.target.go != null)
        {
            Vector3 look = Flat(f.target.go.transform.position - f.logicPos);
            target = Vector3.Lerp(f.logicPos, f.target.go.transform.position, 0.4f);
            dist = 420f;
            defenseYaw = look.sqrMagnitude > 1f ? Mathf.Atan2(look.x, look.z) * Mathf.Rad2Deg : GateCamYaw(f.post);
            return true;
        }
        var m = monsters.FirstOrDefault(x => x.dieAt < 0f);
        if (m == null) return false;
        target = m.go.transform.position;
        dist = 420f;
        defenseYaw = GateCamYaw(m.gate);
        return true;
    }

    // ---------------- 화면 ----------------
    void HpBar(Rect r, string label, float ratio, string text, Color c)
    {
        GUI.Label(new Rect(r.x, r.y, 100, r.height), label, small);
        var track = new Rect(r.x + 102, r.y + 4, r.width - 200, r.height - 8);
        GUI.color = new Color(0, 0, 0, 0.5f);
        GUI.DrawTexture(track, Texture2D.whiteTexture);
        GUI.color = c;
        GUI.DrawTexture(new Rect(track.x, track.y, track.width * Mathf.Clamp01(ratio), track.height), Texture2D.whiteTexture);
        GUI.color = Color.white;
        GUI.Label(new Rect(track.xMax + 6, r.y, 90, r.height), text, small);
    }

    void DefenseGUI()
    {
        GUILayout.Space(6);
        GUILayout.Label("<b>공장 방어</b>", body);
        HpBar(GUILayoutUtility.GetRect(420, 22), "공장 내구도", (float)(factoryHp / FactoryHpMax), $"{factoryHp:0}/{FactoryHpMax:0}", new Color(0.85f, 0.3f, 0.25f));
        int alive = monsters.Count(m => m.dieAt < 0f);
        GUILayout.Label($"습격 {wave}번째 · 괴물 {alive}마리 · 다음 습격 {Mathf.Max(0f, nextWaveAt - Time.time):0}초 뒤 {string.Join("·", nextWaveGates.Select(i => GateName[i]))} 문", small);
        for (int row = 0; row < 2; row++)
        {
            GUILayout.BeginHorizontal();
            for (int k = row * 2; k < row * 2 + 2 && k < gates.Count; k++)
            {
                var gt = gates[k];
                string label = gt.Broken ? $"{GateName[k]} 문 부서짐 · 다시 세우기 60원"
                    : gt.hp < gt.maxHp ? $"{GateName[k]} 문 {gt.hp:0}/{gt.maxHp:0} · 수리 20원" : $"{GateName[k]} 문 {gt.hp:0}/{gt.maxHp:0}";
                if (GUILayout.Button(label, GUILayout.Width(214)) && !gameOver)
                {
                    if (gt.Broken && money >= 60)
                    {
                        money -= 60;
                        gt.hp = gt.maxHp;
                    }
                    else if (!gt.Broken && gt.hp < gt.maxHp && money >= 20)
                    {
                        money -= 20;
                        gt.hp = Mathf.Min(gt.maxHp, gt.hp + 30f);
                    }
                }
            }
            GUILayout.EndHorizontal();
        }
        if (GUILayout.Button($"병정 초파리 {SoldierCost}원 → {GateName[NextPost()]} 문") && money >= SoldierCost && !gameOver)
        {
            money -= SoldierCost;
            soldiersBought++;
            AddSoldier();
        }
        GUILayout.Label("병정도 싸우라고 가르치지 않았어요. 멀리서 작게 움직이는 괴물엔 다가가기 뉴런이 켜져 방아쇠를 당기고(조준은 방향 틀기 뉴런), 가까이 덮치는 거미엔 거대섬유가 켜져 도망쳐요. 총은 게임 소품이에요.", small);
        if (defenseText != "") GUILayout.Label($"<i>{defenseText}</i>", small);
    }

    void SoldierRowsGUI(float w)
    {
        foreach (var f in flies.Where(x => x.soldier))
        {
            GUILayout.Label($"<b>병정 {f.name}</b> (개체 {f.seed}) — {GateName[f.post]} 문 · {f.soldierAction}", body);
            GUILayout.Label($"처치 {f.kills} · 사격 {f.bursts}번(명중 {f.hits}/{f.shots}발) · 도망 {f.flees} · 제자리 {f.holds}", small);
            GUILayout.BeginHorizontal();
            GUILayout.Label($"장비: {GearName[f.gear]} · 몸 속도 ×{f.BodySpeed:0.00} · 한 번에 {3 + f.gear}발", small, GUILayout.Width(260));
            if (f.gear < 3)
            {
                int cost = GearCost[f.gear];
                if (GUILayout.Button($"강화 → {GearName[f.gear + 1]} {cost}원", GUILayout.Width(200)) && money >= cost)
                {
                    money -= cost;
                    f.gear++;
                    ApplyGear(f);
                }
            }
            else GUILayout.Label("최고 단계", small);
            GUILayout.EndHorizontal();
            Bar(GUILayoutUtility.GetRect(w - 40, 22), "도주 거대섬유", MiniJson.Num(f.values, "GF_peak50ms_hz"), GfThreshold, 200, new Color(0.35f, 0.6f, 0.95f));
            Bar(GUILayoutUtility.GetRect(w - 40, 22), "다가가기", MiniJson.Num(f.values, "approach_hz"), SoldierFireHz, 40, new Color(0.9f, 0.45f, 0.3f));
            GUILayout.Label((f.pending ? "뇌 계산 중… " : "왜: ") + f.reason + (f.outcome != "" ? $" → {f.outcome}" : ""), small);
            GUILayout.Space(8);
        }
    }

    void DefenseWorldGUI()
    {
        if (cam == null) return;
        foreach (var gt in gates)
        {
            Vector3 sp = cam.WorldToScreenPoint(GatePoint(gt.index) + Vector3.up * 95f);
            if (sp.z <= 0f) continue;
            var r = new Rect(sp.x - 40f, Screen.height - sp.y, 80f, 9f);
            GUI.color = new Color(0, 0, 0, 0.6f);
            GUI.DrawTexture(r, Texture2D.whiteTexture);
            GUI.color = Color.Lerp(new Color(0.9f, 0.2f, 0.15f), new Color(0.3f, 0.85f, 0.35f), gt.hp / gt.maxHp);
            GUI.DrawTexture(new Rect(r.x, r.y, r.width * Mathf.Clamp01(gt.hp / gt.maxHp), r.height), Texture2D.whiteTexture);
            GUI.color = Color.white;
            if (gt.Broken) GUI.Label(new Rect(r.x - 6, r.y - 20, 120, 20), "부서짐", small);
        }
        foreach (var m in monsters)
        {
            if (m.kind != "spider" || m.dieAt > 0f) continue;
            Vector3 sp = cam.WorldToScreenPoint(m.go.transform.position + Vector3.up * 70f);
            if (sp.z <= 0f) continue;
            var r = new Rect(sp.x - 30f, Screen.height - sp.y, 60f, 6f);
            GUI.color = new Color(0, 0, 0, 0.6f);
            GUI.DrawTexture(r, Texture2D.whiteTexture);
            GUI.color = new Color(0.85f, 0.15f, 0.1f);
            GUI.DrawTexture(new Rect(r.x, r.y, r.width * Mathf.Clamp01(m.hp / m.maxHp), r.height), Texture2D.whiteTexture);
            GUI.color = Color.white;
        }
    }

    void GameOverGUI()
    {
        if (!gameOver) return;
        GUI.color = new Color(0, 0, 0, 0.72f);
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), Texture2D.whiteTexture);
        GUI.color = Color.white;
        float bw = 540, bh = 240;
        GUILayout.BeginArea(new Rect((Screen.width - bw) / 2, (Screen.height - bh) / 2, bw, bh), GUI.skin.box);
        GUILayout.Label("<b>게임 오버</b>", title);
        GUILayout.Label($"괴물이 문을 부수고 공장을 무너뜨렸어요.\n버틴 습격 {wave}번 · 쓰러뜨린 괴물 {totalKills}마리 · 병정 {flies.Count(x => x.soldier)}마리", body);
        GUILayout.Space(10);
        if (GUILayout.Button("다시 시작", GUILayout.Height(42))) SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
        GUILayout.EndArea();
    }
}
