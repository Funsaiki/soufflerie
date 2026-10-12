import type { Sample } from './measurements';

const SERIES = [
  { key: 'cd', label: 'Cx', color: '#f0a050' },
  { key: 'cl', label: 'Cz', color: '#5aa9ff' },
  { key: 'cs', label: 'Cy', color: '#9aa3ad' },
] as const;

/** Time history of the force coefficients, x axis in convective times t* = t U / L. */
export function drawChart(canvas: HTMLCanvasElement, samples: Sample[], stepsToTime: number, span: number): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(canvas.clientWidth * dpr), H = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }
  const g = canvas.getContext('2d')!;
  g.clearRect(0, 0, W, H);
  g.font = `${10 * dpr}px system-ui, sans-serif`;
  if (samples.length < 2) {
    g.fillStyle = '#7d8590';
    g.fillText('En attente de mesures…', 8 * dpr, H / 2);
    return;
  }
  const tEnd = samples[samples.length - 1].step * stepsToTime;
  const tStart = Math.max(samples[0].step * stepsToTime, tEnd - span);
  const vis = samples.filter((s) => s.step * stepsToTime >= tStart);
  let lo = Infinity, hi = -Infinity;
  for (const s of vis) for (const { key } of SERIES) {
    lo = Math.min(lo, s[key]);
    hi = Math.max(hi, s[key]);
  }
  if (!Number.isFinite(lo)) return;
  const pad = Math.max((hi - lo) * 0.1, 0.02);
  lo -= pad;
  hi += pad;
  const L = 34 * dpr, R = 6 * dpr, T = 6 * dpr, B = 16 * dpr;
  const X = (t: number) => L + ((t - tStart) / Math.max(tEnd - tStart, 1e-9)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);

  g.strokeStyle = '#2a3038';
  g.fillStyle = '#7d8590';
  g.lineWidth = 1;
  const ticks = niceTicks(lo, hi, 4);
  for (const v of ticks) {
    g.beginPath();
    g.moveTo(L, Y(v));
    g.lineTo(W - R, Y(v));
    g.stroke();
    g.fillText(v.toFixed(Math.max(0, -Math.floor(Math.log10(ticks[1] - ticks[0] || 1)))), 2 * dpr, Y(v) + 3 * dpr);
  }
  if (lo < 0 && hi > 0) {
    g.strokeStyle = '#3d4550';
    g.beginPath();
    g.moveTo(L, Y(0));
    g.lineTo(W - R, Y(0));
    g.stroke();
  }
  g.fillText(`t* = ${tEnd.toFixed(1)}`, W - R - 60 * dpr, H - 3 * dpr);

  g.lineWidth = 1.5 * dpr;
  for (const { key, color } of SERIES) {
    g.strokeStyle = color;
    g.beginPath();
    vis.forEach((s, i) => {
      const x = X(s.step * stepsToTime), y = Y(s[key]);
      if (i) g.lineTo(x, y);
      else g.moveTo(x, y);
    });
    g.stroke();
  }
}

function niceTicks(lo: number, hi: number, n: number): number[] {
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  return out;
}

export const CHART_LEGEND = SERIES;
