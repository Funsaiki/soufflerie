/** Same colormaps as the WGSL ones, for the legend. */

export function turbo(x: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, x));
  const v = [1, t, t * t, t * t * t];
  const v2 = [v[2] * v[2], v[3] * v[2]];
  const d = (k: number[], k2: number[]) => k[0] * v[0] + k[1] * v[1] + k[2] * v[2] + k[3] * v[3] + k2[0] * v2[0] + k2[1] * v2[1];
  const c = [
    d([0.13572138, 4.6153926, -42.66032258, 132.13108234], [-152.94239396, 59.28637943]),
    d([0.09140261, 2.19418839, 4.84296658, -14.18503333], [4.27729857, 2.82956604]),
    d([0.1066733, 12.64194608, -60.58204836, 110.36276771], [-89.90310912, 27.34824973]),
  ];
  return c.map((x) => Math.min(1, Math.max(0, x))) as [number, number, number];
}

export function diverging(x: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, x));
  const lo = [0.23, 0.299, 0.754], mid = [0.865, 0.865, 0.865], hi = [0.706, 0.016, 0.15];
  const [a, b, s] = t < 0.5 ? [lo, mid, t * 2] : [mid, hi, t * 2 - 1];
  return [0, 1, 2].map((i) => a[i] + (b[i] - a[i]) * s) as [number, number, number];
}

export function drawColorbar(canvas: HTMLCanvasElement, isDiverging: boolean): void {
  const w = 256;
  if (canvas.width !== w) {
    canvas.width = w;
    canvas.height = 1;
  }
  const g = canvas.getContext('2d')!;
  const img = g.createImageData(w, 1);
  for (let i = 0; i < w; i++) {
    const c = (isDiverging ? diverging : turbo)(i / (w - 1));
    img.data.set([c[0] * 255, c[1] * 255, c[2] * 255, 255], i * 4);
  }
  g.putImageData(img, 0, 0);
}
