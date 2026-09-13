// 뇌 판정 근거 표시 — 허브의 brain 메시지(보상 PAM·처벌 PPL1·도주 GF 수치와 문턱)를 막대로 보여 준다.
// 막대 가운데 세로선 = 문턱(지수 1.0). 막대 끝 = 지수 2.0.
const $ = (id) => document.getElementById(id);

const style = document.createElement('style');
style.textContent = `
  .brain { display:grid; gap:6px; border-top:1px solid var(--line); padding-top:8px; }
  .meter { display:grid; grid-template-columns:72px 1fr 96px; gap:8px; align-items:center; font-size:12px; }
  .meter span { color:var(--muted); }
  .meter i { position:relative; display:block; height:8px; background:#232a31; border-radius:3px; overflow:hidden; }
  .meter em { position:absolute; inset:0 auto 0 0; width:0; background:var(--eye); transition:width .4s; }
  .meter u { position:absolute; top:0; bottom:0; left:50%; width:2px; background:var(--ink); opacity:.75; }
  .meter b { font-size:11.5px; text-align:right; font-weight:500; }
  #m-punish { background:#9c85f2; } #m-escape { background:#5ba8e8; }
  .why { font-size:11.5px; color:var(--muted); line-height:1.45; }
  @media (prefers-reduced-motion: reduce) { .meter em { transition:none; } }
`;
document.head.appendChild(style);

const VERDICT = {
  like: '뇌가 강하게 반응 → 좋아요',
  avoid: '회피 반응 → 바로 넘김',
  neutral: '반응 약함 → 조금 보고 넘김',
  error: '뇌 계산 실패',
  random: '무작위 규칙',
};

function setMeter(key, value, thr, unit) {
  const idx = thr ? value / thr : null;
  $('m-' + key).style.width = idx == null ? '0%' : `${Math.min(100, idx * 50).toFixed(1)}%`;
  $('v-' + key).textContent = value == null ? '—' : `${value}${unit} / ${thr ?? '—'}`;
}

function clear(text) {
  $('h-verdict').textContent = text;
  $('h-reason').textContent = '';
  for (const k of ['reward', 'punish', 'escape']) { $('m-' + k).style.width = '0%'; $('v-' + k).textContent = '—'; }
}

let currentReel = null;
window.onFlyMessage = (m) => {
  if (m.type === 'hello') {
    const brain = m.decider !== 'random';
    $('h-brain').hidden = !brain;
    $('h-note').textContent = brain
      ? '판단: Shiu 2024 초파리 전뇌 모델. 릴스 첫 1초를 겹눈으로 보여 주고, 보상 도파민 뉴런(PAM)이 문턱을 넘으면 좋아요, 거대섬유나 처벌 도파민 뉴런(PPL1)이 넘으면 바로 넘겨요. 초파리는 영상 내용이 아니라 밝기와 움직임에 반응해요.'
      : '판단: 비교용 무작위 규칙이에요. 뇌 모델을 쓰지 않아요.';
  } else if (m.type === 'reel' && m.id !== currentReel) {
    currentReel = m.id;
    clear('뇌가 보는 중…');
  } else if (m.type === 'brain' && m.id === currentReel) {
    if (m.status === 'capturing') clear('겹눈으로 1초 보는 중');
    else if (m.status === 'computing') clear('뇌 시뮬레이션 중(수 초)');
    else if (m.status === 'done') {
      $('h-verdict').textContent = VERDICT[m.verdict] || m.verdict;
      $('h-reason').textContent = m.reason || '';
      const v = m.values || {}, t = m.thresholds || {};
      setMeter('reward', v.PAM_mean_hz, t.PAM_mean_hz, 'Hz');
      setMeter('approach', v.approach_hz, t.approach_hz, 'Hz');
      setMeter('punish', v.PPL1_mean_hz, t.PPL1_mean_hz, 'Hz');
      setMeter('escape', v.GF_peak50ms_hz, t.GF_peak50ms_hz, 'Hz');
    }
  }
};
