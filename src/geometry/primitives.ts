import type { V3 } from '../render/math';

/**
 * Built-in test bodies. All are Z-up, in metres, with the front facing -x (towards the inlet).
 * Bodies that stand on the ground have their lowest point at z = 0.
 */

type Tri = number[];

function pushQuad(out: Tri, a: V3, b: V3, c: V3, d: V3): void {
  out.push(...a, ...b, ...c, ...a, ...c, ...d);
}

export function sphere(diameter = 1, seg = 96): Float32Array {
  const r = diameter / 2;
  const out: Tri = [];
  const rings = seg / 2;
  const p = (i: number, j: number): V3 => {
    const th = (Math.PI * i) / rings, ph = (2 * Math.PI * j) / seg;
    return [r * Math.sin(th) * Math.cos(ph), r * Math.sin(th) * Math.sin(ph), r * Math.cos(th)];
  };
  for (let i = 0; i < rings; i++)
    for (let j = 0; j < seg; j++) pushQuad(out, p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
  return new Float32Array(out);
}

/** Cylinder with its axis along z (cross-flow), like the classic von Kármán experiment. */
export function cylinder(diameter = 1, height = 4, seg = 96): Float32Array {
  const r = diameter / 2;
  const out: Tri = [];
  const h = height / 2;
  for (let j = 0; j < seg; j++) {
    const a0 = (2 * Math.PI * j) / seg, a1 = (2 * Math.PI * (j + 1)) / seg;
    const p0: V3 = [r * Math.cos(a0), r * Math.sin(a0), 0];
    const p1: V3 = [r * Math.cos(a1), r * Math.sin(a1), 0];
    pushQuad(out, [p0[0], p0[1], -h], [p1[0], p1[1], -h], [p1[0], p1[1], h], [p0[0], p0[1], h]);
    out.push(0, 0, h, p0[0], p0[1], h, p1[0], p1[1], h);
    out.push(0, 0, -h, p1[0], p1[1], -h, p0[0], p0[1], -h);
  }
  return new Float32Array(out);
}

export function box(lx = 1, ly = 1, lz = 1): Float32Array {
  const x = lx / 2, y = ly / 2, z = lz / 2;
  const out: Tri = [];
  const v = (i: number): V3 => [i & 1 ? x : -x, i & 2 ? y : -y, i & 4 ? z : -z];
  const faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
  for (const f of faces) pushQuad(out, v(f[0]), v(f[1]), v(f[2]), v(f[3]));
  return new Float32Array(out);
}

/**
 * Finite wing with a NACA 4-digit section: chord 1 along x, span along y, closed tips.
 * The trailing edge is closed (modified last coefficient) so the solid is watertight.
 */
export function nacaWing(code: string, chord = 1, span = 4, n = 120): Float32Array {
  const m = +code[0] / 100, p = +code[1] / 10, t = +code.slice(2) / 100;
  const upper: [number, number][] = [];
  const lower: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
    let yc = 0, dyc = 0;
    if (m > 0) {
      if (x < p) {
        yc = (m / (p * p)) * (2 * p * x - x * x);
        dyc = ((2 * m) / (p * p)) * (p - x);
      } else {
        yc = (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x * x);
        dyc = ((2 * m) / (1 - p) ** 2) * (p - x);
      }
    }
    const th = Math.atan(dyc);
    upper.push([(x - yt * Math.sin(th) - 0.5) * chord, (yc + yt * Math.cos(th)) * chord]);
    lower.push([(x + yt * Math.sin(th) - 0.5) * chord, (yc - yt * Math.cos(th)) * chord]);
  }
  const out: Tri = [];
  const s = span / 2;
  const P = (q: [number, number], y: number): V3 => [q[0], y, q[1]];
  for (let i = 0; i < n; i++) {
    pushQuad(out, P(upper[i], -s), P(upper[i + 1], -s), P(upper[i + 1], s), P(upper[i], s));
    pushQuad(out, P(lower[i], s), P(lower[i + 1], s), P(lower[i + 1], -s), P(lower[i], -s));
    pushQuad(out, P(upper[i], s), P(upper[i + 1], s), P(lower[i + 1], s), P(lower[i], s));
    pushQuad(out, P(lower[i], -s), P(lower[i + 1], -s), P(upper[i + 1], -s), P(upper[i], -s));
  }
  return new Float32Array(out);
}

// ---------------------------------------------------------------- implicit bodies (SDF + surface nets)

type Sdf = (x: number, y: number, z: number) => number;

/** Naive surface nets: one vertex per sign-changing cell, one quad per sign-changing edge. */
export function surfaceNets(sdf: Sdf, min: V3, max: V3, res: number): Float32Array {
  const ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const h = Math.max(...ext) / res;
  const dims = ext.map((e) => Math.ceil(e / h) + 3);
  const [nx, ny, nz] = dims;
  const o: V3 = [min[0] - h, min[1] - h, min[2] - h];
  const val = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) val[i + nx * (j + ny * k)] = sdf(o[0] + i * h, o[1] + j * h, o[2] + k * h);

  const cx = nx - 1, cy = ny - 1, cz = nz - 1;
  const vid = new Int32Array(cx * cy * cz).fill(-1);
  const verts: number[] = [];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const corner = new Float64Array(8);
  for (let k = 0; k < cz; k++)
    for (let j = 0; j < cy; j++)
      for (let i = 0; i < cx; i++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const v = val[(i + (c & 1)) + nx * ((j + ((c >> 1) & 1)) + ny * (k + ((c >> 2) & 1)))];
          corner[c] = v;
          if (v < 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        let sx = 0, sy = 0, sz = 0, cnt = 0;
        for (const [a, b] of edges) {
          if ((corner[a] < 0) === (corner[b] < 0)) continue;
          const t = corner[a] / (corner[a] - corner[b]);
          sx += (a & 1) + t * ((b & 1) - (a & 1));
          sy += ((a >> 1) & 1) + t * (((b >> 1) & 1) - ((a >> 1) & 1));
          sz += ((a >> 2) & 1) + t * (((b >> 2) & 1) - ((a >> 2) & 1));
          cnt++;
        }
        vid[i + cx * (j + cy * k)] = verts.length / 3;
        verts.push(o[0] + (i + sx / cnt) * h, o[1] + (j + sy / cnt) * h, o[2] + (k + sz / cnt) * h);
      }

  const out: Tri = [];
  const V = (i: number, j: number, k: number): V3 => {
    const id = vid[i + cx * (j + cy * k)] * 3;
    return [verts[id], verts[id + 1], verts[id + 2]];
  };
  for (let k = 1; k < nz - 1; k++)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const v0 = val[i + nx * (j + ny * k)] < 0;
        // edge along x to (i+1, j, k)
        if (i < cx && v0 !== val[i + 1 + nx * (j + ny * k)] < 0) {
          const q: V3[] = [V(i, j - 1, k - 1), V(i, j, k - 1), V(i, j, k), V(i, j - 1, k)];
          if (v0) q.reverse();
          pushQuad(out, q[0], q[1], q[2], q[3]);
        }
        if (j < cy && v0 !== val[i + nx * (j + 1 + ny * k)] < 0) {
          const q: V3[] = [V(i - 1, j, k - 1), V(i - 1, j, k), V(i, j, k), V(i, j, k - 1)];
          if (v0) q.reverse();
          pushQuad(out, q[0], q[1], q[2], q[3]);
        }
        if (k < cz && v0 !== val[i + nx * (j + ny * (k + 1))] < 0) {
          const q: V3[] = [V(i - 1, j - 1, k), V(i, j - 1, k), V(i, j, k), V(i - 1, j, k)];
          if (v0) q.reverse();
          pushQuad(out, q[0], q[1], q[2], q[3]);
        }
      }
  return new Float32Array(out);
}

/** 2D rectangle [0, len] x [-half, half] whose two corners at u = 0 are rounded with radius r. */
function frontRoundedRect(u: number, v: number, len: number, half: number, r: number): number {
  const ext = r + 0.01 * len; // extend backwards so the rear corners stay sharp after the cut
  const cu = (len + ext) / 2, hu = (len + ext) / 2;
  const qx = Math.abs(u - cu) - (hu - r), qy = Math.abs(v) - (half - r);
  const d = Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
  return Math.max(d, u - len);
}

function verticalCylinder(x: number, y: number, z: number, cx: number, cy: number, r: number, z0: number, z1: number): number {
  const dr = Math.hypot(x - cx, y - cy) - r;
  const dz = Math.max(z0 - z, z - z1);
  return Math.min(Math.max(dr, dz), 0) + Math.hypot(Math.max(dr, 0), Math.max(dz, 0));
}

/**
 * Ahmed body (Ahmed, Ramm & Faltin 1984): 1044 x 389 x 288 mm, 100 mm front radii,
 * 222 mm rear slant at 25°, 50 mm ground clearance on four cylindrical stilts.
 */
export function ahmedBody(slantDeg = 25): Float32Array {
  const L = 1.044, W = 0.389, H = 0.288, r = 0.1, clear = 0.05;
  const slantLen = 0.222;
  const a = (slantDeg * Math.PI) / 180;
  const zTop = clear + H;
  const p0x = L - slantLen * Math.cos(a);
  const nx = Math.sin(a), nz = Math.cos(a);
  const legs: [number, number][] = [[0.163, 0.1175], [0.163, -0.1175], [0.633, 0.1175], [0.633, -0.1175]];
  const sdf: Sdf = (x, y, z) => {
    const top = frontRoundedRect(x, y, L, W / 2, r);
    const side = frontRoundedRect(x, z - (clear + H / 2), L, H / 2, r);
    let d = Math.max(top, side);
    d = Math.max(d, (x - p0x) * nx + (z - zTop) * nz);
    for (const [lx, ly] of legs) d = Math.min(d, verticalCylinder(x, y, z, lx, ly, 0.015, -0.01, clear + 0.02));
    return d;
  };
  return surfaceNets(sdf, [-0.02, -W / 2 - 0.02, -0.01], [L + 0.02, W / 2 + 0.02, zTop + 0.02], 300);
}

function roundBox(x: number, y: number, z: number, c: V3, half: V3, r: number): number {
  const qx = Math.abs(x - c[0]) - (half[0] - r);
  const qy = Math.abs(y - c[1]) - (half[1] - r);
  const qz = Math.abs(z - c[2]) - (half[2] - r);
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0) - r;
}

function smoothMin(a: number, b: number, k: number): number {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - (h * h * k) / 4;
}

/** Generic hatchback-like car, 4.2 m long, wheels touching the ground at z = 0. */
export function simpleCar(): Float32Array {
  const wheels: V3[] = [[0.85, 0.74, 0.33], [0.85, -0.74, 0.33], [3.35, 0.74, 0.33], [3.35, -0.74, 0.33]];
  const sdf: Sdf = (x, y, z) => {
    const body = roundBox(x, y, z, [2.1, 0, 0.6], [2.1, 0.88, 0.38], 0.22);
    const cabin = roundBox(x, y, z, [2.45, 0, 1.08], [1.2, 0.76, 0.36], 0.3);
    let d = smoothMin(body, cabin, 0.35);
    // windscreen rake: cut the front of the cabin with an inclined plane
    const a = 0.95;
    const windscreen = (x - 1.25) * -Math.sin(a) + (z - 0.95) * Math.cos(a);
    d = Math.max(d, Math.min(windscreen, z - 0.95));
    // wheel arches and wheels (cylinders along y)
    for (const w of wheels) {
      const dr = Math.hypot(x - w[0], z - w[2]);
      const arch = Math.max(dr - 0.39, Math.abs(y - w[1]) - 0.16);
      d = Math.max(d, -arch);
      const wheel = Math.max(dr - 0.34, Math.abs(y - w[1]) - 0.12);
      d = Math.min(d, wheel);
    }
    return d;
  };
  return surfaceNets(sdf, [-0.05, -1.0, -0.02], [4.25, 1.0, 1.5], 260);
}
