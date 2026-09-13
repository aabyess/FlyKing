// FlyKing 라이브 뷰어 — 초파리가 폰으로 릴스를 보고 앞다리로 넘기고 좋아요를 누른다.
// 좌표: glTF(three.js)는 Y 위. Blender (x, y, z) → (x, z, -y). 초파리는 +X(폰 쪽)를 보고, 오른쪽 다리가 +Z 쪽이다.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const V3 = THREE.Vector3, Q = THREE.Quaternion;
const $ = (id) => document.getElementById(id);
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// ---------- 조정값 ----------
const REAR_DEG = 14;        // 폰 볼 때 몸을 세우는 각(뒷다리 축)
const REACH_X = 13.0;       // 하트까지 앞다리 밑동에서 떨어질 가로 거리(장면 단위, 초파리 몸길이 25)
const HEAD_CLEAR = 1.5;     // 머리와 화면 사이 최소 간격
const WALK_S = 3.2, RISE_S = 0.9;
const TIP_LEN = 0.084;      // Tarsus5 뼈 길이(뼈 로컬 단위 = 실제 mm)
const UI = { heart: [0.88, 0.66], swipeFrom: [0.5, 0.86], swipeTo: [0.5, 0.52], restL: [0.28, 0.975], restR: [0.72, 0.975] };

// ---------- 렌더러·장면 ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.AgXToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
$('stage').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171c);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 3000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x3a2a1e, 0.4));
const sun = new THREE.DirectionalLight(0xffe7cc, 2.4);
sun.position.set(-40, 75, 50);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -70, right: 70, top: 70, bottom: -70, near: 1, far: 300 });
sun.shadow.bias = -0.0004;
scene.add(sun);
const glow = new THREE.PointLight(0xffffff, 0, 90, 1);
scene.add(glow);

const table = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), new THREE.MeshStandardMaterial({ color: 0x4b3627, roughness: 0.78 }));
table.rotation.x = -Math.PI / 2;
table.receiveShadow = true;
scene.add(table);

// ---------- 폰 화면(캔버스 텍스처) ----------
const SW = 540, SH = 1141;   // 표시 영역 비율 68.4 : 144.6
const scr = document.createElement('canvas');
scr.width = SW; scr.height = SH;
const sg = scr.getContext('2d');
const screenTex = new THREE.CanvasTexture(scr);
screenTex.colorSpace = THREE.SRGBColorSpace;
screenTex.flipY = false;     // glTF UV는 위가 v=0
screenTex.anisotropy = 8;

const state = {
  connected: false, source: '—', fake: true, reel: null, frame: null, liked: false, likePop: 0,
  ripple: null, drag: 0, slideIn: 0, progress: null, seen: 0, likedTotal: 0, reelStart: performance.now(),
  phase: 'walk', phaseT: 0, action: null, queue: [], frozen: null, ws: null,
};

function heartPath(g, cx, cy, s) {
  g.beginPath();
  g.moveTo(cx, cy + 0.62 * s);
  g.bezierCurveTo(cx - 1.15 * s, cy - 0.1 * s, cx - 0.6 * s, cy - 0.95 * s, cx, cy - 0.42 * s);
  g.bezierCurveTo(cx + 0.6 * s, cy - 0.95 * s, cx + 1.15 * s, cy - 0.1 * s, cx, cy + 0.62 * s);
  g.closePath();
}

function wrapLines(g, text, maxW, maxLines) {
  const out = []; let line = '';
  for (const ch of text || '') {
    if (ch === '\n') { out.push(line); line = ''; if (out.length === maxLines) break; continue; }
    if (g.measureText(line + ch).width > maxW) { out.push(line); line = ch; if (out.length === maxLines) break; } else line += ch;
  }
  if (out.length < maxLines && line) out.push(line);
  if (out.length === maxLines && (text || '').length > out.join('').length) out[maxLines - 1] = out[maxLines - 1].slice(0, -1) + '…';
  return out;
}

function drawReelLayer(offsetY) {
  if (state.frame) {
    const f = state.frame, s = Math.max(SW / f.width, SH / f.height);
    sg.drawImage(f, (SW - f.width * s) / 2, (SH - f.height * s) / 2 + offsetY, f.width * s, f.height * s);
  } else {
    const gr = sg.createLinearGradient(0, offsetY, 0, SH + offsetY);
    gr.addColorStop(0, '#f09a5a'); gr.addColorStop(1, '#3a1d6e');
    sg.fillStyle = gr; sg.fillRect(0, offsetY, SW, SH);
    sg.fillStyle = 'rgba(255,255,255,.85)'; sg.font = '600 30px -apple-system, "Apple SD Gothic Neo", sans-serif';
    sg.textAlign = 'center'; sg.fillText(state.connected ? '릴스 기다리는 중' : '허브 연결 기다리는 중', SW / 2, SH / 2 + offsetY);
    sg.textAlign = 'left';
  }
}

function drawScreen(now) {
  sg.fillStyle = '#000'; sg.fillRect(0, 0, SW, SH);
  const offset = (-state.drag + state.slideIn) * SH;
  drawReelLayer(offset);
  let g = sg.createLinearGradient(0, 0, 0, 220); g.addColorStop(0, 'rgba(0,0,0,.45)'); g.addColorStop(1, 'rgba(0,0,0,0)');
  sg.fillStyle = g; sg.fillRect(0, 0, SW, 220);
  g = sg.createLinearGradient(0, SH - 360, 0, SH); g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,.6)');
  sg.fillStyle = g; sg.fillRect(0, SH - 360, SW, 360);

  const font = (w, px) => `${w} ${px}px -apple-system, "Apple SD Gothic Neo", sans-serif`;
  sg.fillStyle = '#fff'; sg.font = font(700, 38); sg.fillText('릴스', 30, 84);

  // 오른쪽 버튼 줄
  const hx = UI.heart[0] * SW, hy = UI.heart[1] * SH;
  const pop = 1 + 0.35 * Math.sin(Math.PI * Math.min(1, state.likePop));
  if (state.liked) { heartPath(sg, hx, hy, 30 * pop); sg.fillStyle = '#ff3040'; sg.fill(); }
  else { heartPath(sg, hx, hy, 30); sg.lineWidth = 5; sg.strokeStyle = '#fff'; sg.stroke(); }
  sg.fillStyle = '#fff'; sg.font = font(600, 22); sg.textAlign = 'center';
  sg.fillText(state.reel?.likeCount || '', hx, hy + 62);
  sg.lineWidth = 5; sg.strokeStyle = '#fff';
  sg.beginPath(); sg.arc(hx, hy + 120, 24, 0.25 * Math.PI, 2.1 * Math.PI); sg.lineTo(hx + 26, hy + 146); sg.closePath(); sg.stroke();
  sg.beginPath(); sg.moveTo(hx - 26, hy + 205); sg.lineTo(hx + 26, hy + 185); sg.lineTo(hx + 4, hy + 232); sg.lineTo(hx - 4, hy + 212); sg.closePath(); sg.stroke();
  sg.fillStyle = '#fff'; for (let i = -1; i <= 1; i++) { sg.beginPath(); sg.arc(hx + i * 12, hy + 290, 4.5, 0, 7); sg.fill(); }
  if (state.liked && state.fake) { sg.font = font(600, 18); sg.fillStyle = 'rgba(255,255,255,.8)'; sg.fillText('시험', hx, hy - 44); }
  sg.textAlign = 'left';

  // 작성자·캡션
  const by = SH - 150;
  sg.fillStyle = 'rgba(255,255,255,.9)'; sg.beginPath(); sg.arc(58, by, 24, 0, 7); sg.fill();
  sg.fillStyle = '#fff'; sg.font = font(700, 27); sg.fillText(state.reel?.author || '', 94, by + 9);
  sg.font = font(400, 23); sg.fillStyle = 'rgba(255,255,255,.92)';
  wrapLines(sg, state.reel?.caption || '', SW * 0.74, 2).forEach((l, i) => sg.fillText(l, 34, by + 56 + i * 32));

  // 진행 막대
  const p = state.progress && state.progress.d ? Math.min(1, state.progress.t / state.progress.d) : 0;
  sg.fillStyle = 'rgba(255,255,255,.3)'; sg.fillRect(0, SH - 5, SW, 5);
  sg.fillStyle = '#fff'; sg.fillRect(0, SH - 5, SW * p, 5);

  // 다리가 닿은 자리 물결
  if (state.ripple) {
    const age = (now - state.ripple.t) / 600;
    if (age < 1) {
      sg.strokeStyle = `rgba(255,255,255,${0.9 * (1 - age)})`; sg.lineWidth = 6;
      sg.beginPath(); sg.arc(state.ripple.u * SW, state.ripple.v * SH, 18 + age * 70, 0, 7); sg.stroke();
    } else state.ripple = null;
  }
  screenTex.needsUpdate = true;
}

const glowProbe = document.createElement('canvas');
glowProbe.width = glowProbe.height = 6;
const gp = glowProbe.getContext('2d', { willReadFrequently: true });
function updateGlow(bmp) {
  gp.drawImage(bmp, 0, 0, 6, 6);
  const d = gp.getImageData(0, 0, 6, 6).data;
  let r = 0, g = 0, b = 0;
  for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
  const n = d.length / 4 * 255;
  glow.color.setRGB(r / n, g / n, b / n, THREE.SRGBColorSpace);
}

// ---------- 허브 연결 ----------
function connect() {
  if (!location.host) return;
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'blob';
  state.ws = ws;
  ws.onopen = () => { state.connected = true; };
  ws.onclose = () => { state.connected = false; setTimeout(connect, 1500); };
  ws.onmessage = async (ev) => {
    if (typeof ev.data !== 'string') {
      const bmp = await createImageBitmap(ev.data);
      if (state.frame && state.frame.close) state.frame.close();
      state.frame = bmp; updateGlow(bmp);
      return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === 'hello') { state.source = m.source; state.fake = m.like !== 'real'; }
    else if (m.type === 'reel') {
      if (!state.reel || state.reel.id !== m.id) {
        state.slideIn = state.drag > 0 ? 0.32 : 0; state.drag = 0; state.liked = !!m.liked;
        state.reelStart = performance.now(); state.progress = null;
      }
      state.reel = m; state.seen = m.seen; state.likedTotal = m.likedTotal;
    } else if (m.type === 'act') state.queue.push(m);
    else if (m.type === 'liked') {
      if (state.reel && m.id === state.reel.id) { state.liked = m.liked; state.likePop = 0.001; }
      state.likedTotal = m.likedTotal;
    } else if (m.type === 'progress') state.progress = m;
  };
}
function sendTouch(action, id) {
  if (state.ws && state.ws.readyState === 1 && id) state.ws.send(JSON.stringify({ type: 'touch', action, id }));
}

// ---------- 초파리 뼈대·IK ----------
const fly = {};
const _a = new V3(), _b = new V3(), _c = new V3(), _q1 = new Q(), _q2 = new Q(), _q3 = new Q();
const WEIGHT = [0.35, 1.0, 1.0, 0.6, 0.3];

function tipWorld(leg, out) { return leg.tip.localToWorld(out.set(0, TIP_LEN, 0)); }

function solveLeg(leg, target, iters = 14) {
  for (let it = 0; it < iters; it++) {
    for (let i = leg.chain.length - 1; i >= 0; i--) {
      const j = leg.chain[i];
      const jp = j.getWorldPosition(_b);
      const toEff = tipWorld(leg, _a).sub(jp).normalize();
      const toTgt = _c.copy(target).sub(jp).normalize();
      if (toEff.dot(toTgt) > 0.99999) continue;
      _q1.setFromUnitVectors(toEff, toTgt);
      _q3.identity().slerp(_q1, WEIGHT[i]);
      j.getWorldQuaternion(_q2);
      _q3.multiply(_q2);                       // 새 월드 회전
      j.parent.getWorldQuaternion(_q2);
      j.quaternion.copy(_q2.invert().multiply(_q3));
      j.updateMatrixWorld(true);
    }
    if (tipWorld(leg, _a).distanceTo(target) < 0.05) break;
  }
}

function applyRoot(adv, theta) {
  _q1.setFromAxisAngle(new V3(0, 0, 1), theta);
  const p = fly.base.pos.clone().sub(fly.pivot).applyQuaternion(_q1).add(fly.pivot);
  p.x += adv;
  fly.root.position.copy(p);
  fly.root.quaternion.copy(_q1).multiply(fly.base.quat);
  fly.root.updateMatrixWorld(true);
}

function resetBones() { for (const [b, q] of fly.restQ) b.quaternion.copy(q); }

function makeUvMapper(mesh) {
  const pos = mesh.geometry.attributes.position, uv = mesh.geometry.attributes.uv;
  let iA = 0, iB = 0, iC = 0;
  for (let i = 0; i < uv.count; i++) {
    const u = uv.getX(i), v = uv.getY(i);
    if (u + v < uv.getX(iA) + uv.getY(iA)) iA = i;
    if (u - v > uv.getX(iB) - uv.getY(iB)) iB = i;
    if (v - u > uv.getY(iC) - uv.getX(iC)) iC = i;
  }
  const P = (i) => new V3().fromBufferAttribute(pos, i), U = (i) => new THREE.Vector2().fromBufferAttribute(uv, i);
  const pa = P(iA), ua = U(iA), d1 = P(iB).sub(pa), d2 = P(iC).sub(pa), ub = U(iB).sub(ua), uc = U(iC).sub(ua);
  const det = ub.x * uc.y - uc.x * ub.y;
  const i11 = uc.y / det, i12 = -uc.x / det, i21 = -ub.y / det, i22 = ub.x / det;
  const Au = d1.clone().multiplyScalar(i11).add(d2.clone().multiplyScalar(i21));
  const Av = d1.clone().multiplyScalar(i12).add(d2.clone().multiplyScalar(i22));
  const O = pa.clone().sub(Au.clone().multiplyScalar(ua.x)).sub(Av.clone().multiplyScalar(ua.y));
  const toWorld = (u, v, out = new V3()) => mesh.localToWorld(out.copy(O).addScaledVector(Au, u).addScaledVector(Av, v));
  const n = toWorld(1, 0).sub(toWorld(0, 0)).cross(toWorld(0, 1).sub(toWorld(0, 0))).normalize();
  return { toWorld, normal: n };
}

function screenPoint([u, v], lift, out = new V3()) {
  return fly.screen.toWorld(u, v, out).addScaledVector(fly.screen.normal, lift);
}

// 동작 키프레임: 시간(초), 화면 좌표, 화면에서 뗀 거리
function actionTrack(action) {
  if (action === 'like') return {
    leg: 'RF', rest: UI.restR, touchAt: 0.95, end: 2.3,
    keys: [[0, null, 1.2], [0.7, UI.heart, 2.8], [0.95, UI.heart, 0.12], [1.2, UI.heart, 0.12], [1.65, UI.heart, 3.2], [2.3, null, 1.2]],
  };
  return {
    leg: 'LF', rest: UI.restL, touchAt: 1.65, end: 2.7, drag: [0.95, 1.65],
    keys: [[0, null, 1.2], [0.7, UI.swipeFrom, 2.8], [0.95, UI.swipeFrom, 0.12], [1.65, UI.swipeTo, 0.12], [2.05, UI.swipeTo, 3.2], [2.7, null, 1.2]],
  };
}

function trackTarget(track, t, out) {
  const k = track.keys;
  let i = 0;
  while (i < k.length - 2 && t > k[i + 1][0]) i++;
  const [t0, uv0, l0] = k[i], [t1, uv1, l1] = k[i + 1];
  const s = smooth(t0, t1, t);
  const a = uv0 || track.rest, b = uv1 || track.rest;
  return screenPoint([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s], l0 + (l1 - l0) * s, out);
}

function footAdvance(adv, group) {
  const STEP = 4.0, phase = group ? STEP / 2 : 0;
  const x = (adv + phase) / STEP, k = Math.floor(x), f = x - k, s = smooth(0, 0.35, f);
  return { dx: STEP * (k + s) - phase, lift: Math.sin(Math.PI * s) * 1.6 };
}

const GROUP = { LF: 0, RM: 0, LH: 0, RF: 1, LM: 1, RH: 1 };
const _t = new V3();

function poseFly(now, dt) {
  resetBones();
  const walkK = state.phase === 'walk' ? smooth(0, 1, state.phaseT / WALK_S) : 1;
  const adv = fly.adv * walkK;
  const riseK = state.phase === 'walk' ? 0 : state.phase === 'rise' ? smooth(0, 1, state.phaseT / RISE_S) : 1;
  const theta = fly.theta * riseK + Math.sin(now / 900) * 0.006 * riseK;
  applyRoot(adv, theta);

  for (const L of ['LM', 'RM', 'LH', 'RH']) {
    const leg = fly.legs[L];
    const st = walkK < 1 ? footAdvance(adv, GROUP[L]) : { dx: adv, lift: 0 };
    _t.copy(leg.restFoot); _t.x += Math.min(st.dx, fly.adv); _t.y += st.lift;
    if (L[1] === 'M') _t.x -= 1.6 * riseK;
    solveLeg(leg, _t);
  }
  for (const L of ['LF', 'RF']) {
    const leg = fly.legs[L];
    const st = walkK < 1 ? footAdvance(adv, GROUP[L]) : { dx: adv, lift: 0 };
    const ground = _t.copy(leg.restFoot); ground.x += Math.min(st.dx, fly.adv); ground.y += st.lift;
    const rest = screenPoint(L === 'LF' ? UI.restL : UI.restR, 1.2, new V3());
    let target = ground.lerp(rest, riseK);
    const act = state.action;
    if (act && act.track.leg === L) target = trackTarget(act.track, act.t, new V3());
    solveLeg(leg, target);
  }
}

function stepActions(dt) {
  if (state.phase !== 'watch') return;
  if (!state.action && state.queue.length) {
    const m = state.queue.shift();
    state.action = { m, track: actionTrack(m.action), t: 0, fired: false };
  }
  const a = state.action;
  if (!a) return;
  a.t = state.frozen ? state.frozen.t : a.t + dt;
  const tr = a.track;
  if (tr.drag) state.drag = 0.32 * smooth(tr.drag[0], tr.drag[1], a.t) * (a.fired && a.t > tr.drag[1] + 0.05 ? 1 : 1);
  if (!a.fired && a.t >= tr.touchAt) {
    a.fired = true;
    const uv = a.m.action === 'like' ? UI.heart : UI.swipeTo;
    state.ripple = { u: uv[0], v: uv[1], t: performance.now() };
    if (a.m.action === 'like' && !state.fake) state.likePop = 0.001;
    sendTouch(a.m.action, a.m.id);
  }
  if (a.t >= tr.end && !state.frozen) state.action = null;
}

// ---------- 카메라 ----------
const cams = {};
function setCam(name) {
  const c = cams[name]; if (!c) return;
  camera.position.copy(c.pos); controls.target.copy(c.target); controls.update();
}
['wide', 'shoulder', 'side'].forEach((n, i) => {
  $('cam-' + n).addEventListener('click', () => setCam(n));
  addEventListener('keydown', (e) => { if (e.key === String(i + 1)) setCam(n); });
});

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// ---------- HUD ----------
let hudT = 0;
function updateHud(now) {
  if (now - hudT < 200) return; hudT = now;
  $('dot').classList.toggle('on', state.connected);
  $('h-conn').textContent = state.connected ? `${state.source === 'mock' ? '시험용 가짜 릴스' : '인스타'} · 좋아요 ${state.fake ? '가짜(표시만)' : '실제'}` : '허브 기다리는 중';
  const r = state.reel;
  $('h-author').textContent = r ? (r.author || '작성자 모름') : '—';
  const watched = r ? (now - state.reelStart) / 1000 : 0;
  $('h-time').textContent = r ? `${watched.toFixed(1)}초 / ${r.watch}초 보고 넘김` : '—';
  $('h-plan').textContent = r ? (r.likeAt != null ? `${r.likeAt}초에 좋아요` : '좋아요 안 누름') : '—';
  const act = state.action?.m.action;
  $('h-act').textContent = state.phase === 'walk' ? '폰 앞으로 걸어가는 중' : state.phase === 'rise' ? '몸을 세우는 중'
    : act === 'like' ? '오른쪽 앞다리로 좋아요 누르는 중' : act === 'swipe' ? '왼쪽 앞다리로 넘기는 중' : '릴스 보는 중';
  $('h-total').textContent = `본 릴스 ${state.seen} · 좋아요 ${state.likedTotal}`;
}

// ---------- 시작 ----------
async function start() {
  resize();
  const gltf = await new GLTFLoader().loadAsync('assets/scene.glb');
  scene.add(gltf.scene);
  gltf.scene.traverse((o) => {
    if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; if (o.isSkinnedMesh) o.frustumCulled = false; }
  });
  const screenMesh = gltf.scene.getObjectByName('폰_화면');
  screenMesh.material = new THREE.MeshBasicMaterial({ map: screenTex, toneMapped: false });
  screenMesh.castShadow = false;
  gltf.scene.updateMatrixWorld(true);

  fly.root = gltf.scene.getObjectByName('초파리_뼈대');
  fly.base = { pos: fly.root.position.clone(), quat: fly.root.quaternion.clone() };
  fly.restQ = new Map();
  fly.root.traverse((o) => { if (o.isBone) fly.restQ.set(o, o.quaternion.clone()); });
  const bone = (n) => { const b = fly.root.getObjectByName(n); if (!b) throw new Error('뼈 없음 ' + n); return b; };
  fly.legs = {};
  for (const L of ['LF', 'RF', 'LM', 'RM', 'LH', 'RH']) {
    const leg = { chain: ['Coxa', 'Femur', 'Tibia', 'Tarsus1', 'Tarsus2'].map((c) => bone(L + c)), tip: bone(L + 'Tarsus5') };
    leg.restFoot = tipWorld(leg, new V3());
    fly.legs[L] = leg;
  }
  fly.screen = makeUvMapper(screenMesh);
  const flyC = new THREE.Box3().setFromObject(fly.root).getCenter(new V3());
  if (fly.screen.normal.dot(flyC.clone().sub(fly.screen.toWorld(0.5, 0.5))) < 0) fly.screen.normal.negate();

  // 몸 세우는 축 = 뒷다리 두 발끝 가운데. 배 끝(A6)이 너무 내려가면 각을 줄인다.
  fly.pivot = fly.legs.LH.restFoot.clone().add(fly.legs.RH.restFoot).multiplyScalar(0.5);
  fly.pivot.y = 0;
  const a6 = bone('A6'), a6Rest = a6.getWorldPosition(new V3()).y;
  let theta = THREE.MathUtils.degToRad(REAR_DEG);
  for (; theta > 0; theta -= THREE.MathUtils.degToRad(1)) {
    applyRoot(0, theta);
    if (a6.getWorldPosition(new V3()).y >= 0.55 * a6Rest) break;
  }
  fly.theta = theta;
  // 앞으로 걸어갈 거리: 오른쪽 앞다리 밑동이 하트에서 REACH_X만큼 떨어지게, 머리는 화면에 안 닿게
  const heart = screenPoint(UI.heart, 0);
  let adv = heart.x - REACH_X - bone('RFCoxa').getWorldPosition(new V3()).x;
  const box = new THREE.Box3().setFromObject(fly.root, true);
  const headFront = new V3(box.max.x, bone('Head').getWorldPosition(new V3()).y, 0);
  const s0 = fly.screen.toWorld(0.5, 0.5);
  const clear = headFront.clone().add(new V3(adv, 0, 0)).sub(s0).dot(fly.screen.normal);
  if (clear < HEAD_CLEAR) adv -= (HEAD_CLEAR - clear) / Math.abs(fly.screen.normal.x);
  fly.adv = adv;
  applyRoot(adv, theta);
  const reach = bone('RFCoxa').getWorldPosition(new V3()).distanceTo(heart);
  applyRoot(0, 0);
  window.fly = { info: { thetaDeg: +THREE.MathUtils.radToDeg(theta).toFixed(1), adv: +adv.toFixed(2), reachToHeart: +reach.toFixed(2), clear: +clear.toFixed(2) } };
  console.log('FLY_INFO', JSON.stringify(window.fly.info));

  // 카메라 자리
  const scrC = fly.screen.toWorld(0.5, 0.5);
  const flyEnd = flyC.clone().add(new V3(adv, 0, 0));
  const mid = flyEnd.clone().lerp(scrC, 0.5);
  cams.wide = { target: mid, pos: mid.clone().add(new V3(-58, 40, 62)) };
  cams.shoulder = { target: scrC.clone().add(new V3(0, 2, 0)), pos: flyEnd.clone().add(new V3(-26, 40, -38)) };   // 왼쪽 어깨 위 — 낮으면 머리가 화면을 가린다
  cams.side = { target: flyEnd.clone().lerp(scrC, 0.4), pos: flyEnd.clone().lerp(scrC, 0.4).add(new V3(2, 8, 48)) };
  setCam('wide');
  glow.position.copy(scrC).addScaledVector(fly.screen.normal, 6);
  glow.intensity = 1.3;

  Object.assign(window.fly, {
    state, setCam,
    // 시험용: 동작을 특정 시각에 멈춰 세운다
    freeze(action, t) { state.phase = 'watch'; state.frozen = { t }; state.action = { m: { action, id: null }, track: actionTrack(action), t, fired: t < 0 }; },
    unfreeze() { state.frozen = null; state.action = null; state.drag = 0; },
    skipIntro() { state.phase = 'watch'; state.phaseT = 0; },
  });
  $('loading').remove();
  connect();

  let last = performance.now();
  renderer.setAnimationLoop(() => {
    const now = performance.now(), dt = Math.min(0.25, (now - last) / 1000); last = now;   // 느린 GPU에서도 동작 시간이 실제 시간을 따라가게
    state.phaseT += dt;
    if (state.phase === 'walk' && state.phaseT >= WALK_S) { state.phase = 'rise'; state.phaseT = 0; }
    if (state.phase === 'rise' && state.phaseT >= RISE_S) { state.phase = 'watch'; state.phaseT = 0; }
    if (state.likePop > 0) state.likePop = Math.min(1.01, state.likePop + dt * 3) >= 1.01 ? 0 : state.likePop + dt * 3;
    if (state.slideIn > 0) state.slideIn = Math.max(0, state.slideIn - dt * 1.2);
    stepActions(dt);
    poseFly(now, dt);
    drawScreen(now);
    controls.update();
    renderer.render(scene, camera);
    updateHud(now);
  });
  window.fly.ready = true;
}

start().catch((e) => { $('loading').textContent = '불러오기 실패: ' + e.message; console.error(e); });
