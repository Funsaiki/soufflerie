import { bounds } from './mesh';
import { computeWallLinks, type WallLinks } from './links';
import type { V3 } from '../render/math';

export const FLAG = { FLUID: 0, SOLID: 1, WALL: 2, FREESTREAM: 3, OUTLET: 4 } as const;

export type GroundMode = 'none' | 'fixed' | 'moving';
export type MarginPreset = 'compact' | 'standard' | 'large' | 'huge';

export interface DomainOptions {
  /** Target number of lattice cells. */
  budget: number;
  margins: MarginPreset;
  ground: GroundMode;
  /** Gap between the ground and the bottom of the body, as a fraction of the body height. */
  clearance: number;
  /** Also mark every cell crossed by the surface (for open or very thin meshes). */
  thicken: boolean;
  /** Interpolated bounce-back (exact wall position) instead of the voxel staircase. */
  interpolated: boolean;
}

export interface Domain {
  nx: number;
  ny: number;
  nz: number;
  flags: Uint32Array<ArrayBuffer>;
  /** Triangles in lattice coordinates (1 unit = 1 cell). */
  tris: Float32Array<ArrayBuffer>;
  /** Model units -> cells. */
  cellsPerUnit: number;
  /** Body length along the flow, in cells (reference length). */
  lengthCells: number;
  frontalArea: number;
  planformArea: number;
  blockage: number;
  solidCells: number;
  /** Fraction of ray columns with an odd number of hits (0 for a closed mesh). */
  openFraction: number;
  links: WallLinks;
  bodyMin: V3;
  bodyMax: V3;
}

/**
 * Margins around the body. Upstream/downstream margins scale with the streamwise size
 * D_long = max(length, D_lat); side margins with the cross-section size D_lat = sqrt(width * height),
 * so that slender bodies (wings, cylinders) are not drowned in a huge domain.
 */
const MARGINS: Record<MarginPreset, { up: number; down: number; side: number }> = {
  compact: { up: 0.7, down: 2.0, side: 0.8 },
  standard: { up: 1.0, down: 3.0, side: 1.25 },
  large: { up: 1.5, down: 5.0, side: 2.0 },
  huge: { up: 3.0, down: 6.0, side: 4.0 },
};

export function buildDomain(model: Float32Array, opt: DomainOptions): Domain {
  const b = bounds(model);
  const size: V3 = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
  const Dlat = Math.sqrt(Math.max(size[1] * size[2], 1e-6 * size[0] * size[0]));
  const Dlong = Math.max(size[0], Dlat);
  const m = MARGINS[opt.margins];
  const ground = opt.ground !== 'none';
  const below = ground ? opt.clearance * size[2] : m.side * Dlat;
  const Lx = m.up * Dlong + size[0] + m.down * Dlong;
  const Ly = size[1] + 2 * m.side * Dlat;
  const Lz = below + size[2] + m.side * Dlat;
  const s = Math.cbrt(opt.budget / (Lx * Ly * Lz));
  const nx = Math.max(16, Math.round(Lx * s) + 2);
  const ny = Math.max(16, Math.round(Ly * s) + 2);
  const nz = Math.max(16, Math.round(Lz * s) + 2);

  const offset: V3 = [
    1 + (m.up * Dlong - b.min[0]) * s,
    ny / 2 - ((b.min[1] + b.max[1]) / 2) * s,
    1 + (below - b.min[2]) * s,
  ];
  const tris = new Float32Array(model.length);
  for (let i = 0; i < model.length; i += 3) {
    tris[i] = model[i] * s + offset[0];
    tris[i + 1] = model[i + 1] * s + offset[1];
    tris[i + 2] = model[i + 2] * s + offset[2];
  }

  const { solid, openFraction } = voxelize(tris, nx, ny, nz, opt.thicken);
  const N = nx * ny * nz;
  const flags = new Uint32Array(N);
  let solidCells = 0;
  const front = new Uint8Array(ny * nz);
  const top = new Uint8Array(nx * ny);
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const n = x + nx * (y + ny * z);
        let f: number = FLAG.FLUID;
        if (solid[n]) f = FLAG.SOLID;
        if (x === nx - 1) f = FLAG.OUTLET;
        if (x === 0 || y === 0 || y === ny - 1 || z === nz - 1 || z === 0) {
          if (x !== nx - 1 || y === 0 || y === ny - 1 || z === 0 || z === nz - 1) f = FLAG.FREESTREAM;
        }
        if (z === 0 && ground) f = FLAG.WALL;
        flags[n] = f;
        if (f === FLAG.SOLID) {
          solidCells++;
          front[y + ny * z] = 1;
          top[x + nx * y] = 1;
        }
      }
  const links = opt.interpolated
    ? computeWallLinks(tris, nx, ny, nz, flags)
    : { q: new Uint32Array(5), cells: 0, links: 0, misses: 0 };
  const frontalArea = front.reduce((a, v) => a + v, 0);
  const planformArea = top.reduce((a, v) => a + v, 0);

  return {
    nx, ny, nz, flags, tris,
    cellsPerUnit: s,
    lengthCells: size[0] * s,
    frontalArea,
    planformArea,
    blockage: frontalArea / ((ny - 2) * (nz - 2)),
    solidCells,
    openFraction,
    links,
    bodyMin: [b.min[0] * s + offset[0], b.min[1] * s + offset[1], b.min[2] * s + offset[2]],
    bodyMax: [b.max[0] * s + offset[0], b.max[1] * s + offset[1], b.max[2] * s + offset[2]],
  };
}

/**
 * Solid voxelization by ray parity along the three axes with a majority vote, sampling at cell
 * centres. A closed mesh gives the exact inside test; the vote makes it tolerant to small holes.
 */
export function voxelize(tris: Float32Array, nx: number, ny: number, nz: number, thicken: boolean): { solid: Uint8Array; openFraction: number } {
  const dims = [nx, ny, nz];
  const N = nx * ny * nz;
  const votes = new Uint8Array(N);
  let odd = 0, columns = 0;
  for (let a = 0; a < 3; a++) {
    const r = parityAxis(tris, a, dims, votes);
    odd += r.odd;
    columns += r.columns;
  }
  const solid = new Uint8Array(N);
  for (let i = 0; i < N; i++) solid[i] = votes[i] >= 2 ? 1 : 0;
  const openFraction = columns ? odd / columns : 0;
  if (thicken) markSurface(tris, nx, ny, nz, solid);
  return { solid, openFraction };
}

function parityAxis(tris: Float32Array, a: number, dims: number[], votes: Uint8Array): { odd: number; columns: number } {
  const b = (a + 1) % 3, c = (a + 2) % 3;
  const na = dims[a], nb = dims[b], nc = dims[c];
  const hits: (number[] | undefined)[] = new Array(nb * nc);
  // Tiny offsets keep sample rays off exact edges and vertices of grid-aligned meshes.
  const eb = 1.2345e-4, ec = 2.7183e-5;
  for (let t = 0; t < tris.length; t += 9) {
    const a0 = tris[t + a], a1 = tris[t + 3 + a], a2 = tris[t + 6 + a];
    const b0 = tris[t + b], b1 = tris[t + 3 + b], b2 = tris[t + 6 + b];
    const c0 = tris[t + c], c1 = tris[t + 3 + c], c2 = tris[t + 6 + c];
    const det = (b1 - b0) * (c2 - c0) - (b2 - b0) * (c1 - c0);
    if (Math.abs(det) < 1e-12) continue;
    const jmin = Math.max(0, Math.ceil(Math.min(b0, b1, b2) - 0.5 - eb));
    const jmax = Math.min(nb - 1, Math.floor(Math.max(b0, b1, b2) - 0.5 - eb));
    const kmin = Math.max(0, Math.ceil(Math.min(c0, c1, c2) - 0.5 - ec));
    const kmax = Math.min(nc - 1, Math.floor(Math.max(c0, c1, c2) - 0.5 - ec));
    for (let k = kmin; k <= kmax; k++) {
      const pc = k + 0.5 + ec;
      for (let j = jmin; j <= jmax; j++) {
        const pb = j + 0.5 + eb;
        const w1 = ((pb - b0) * (c2 - c0) - (b2 - b0) * (pc - c0)) / det;
        const w2 = ((b1 - b0) * (pc - c0) - (pb - b0) * (c1 - c0)) / det;
        if (w1 < 0 || w2 < 0 || w1 + w2 > 1) continue;
        const h = a0 + w1 * (a1 - a0) + w2 * (a2 - a0);
        const id = j + nb * k;
        (hits[id] ??= []).push(h);
      }
    }
  }
  let odd = 0, columns = 0;
  const idx = [0, 0, 0];
  const strides = [1, dims[0], dims[0] * dims[1]];
  for (let k = 0; k < nc; k++)
    for (let j = 0; j < nb; j++) {
      const list = hits[j + nb * k];
      if (!list) continue;
      columns++;
      list.sort((x, y) => x - y);
      if (list.length % 2) odd++;
      idx[b] = j;
      idx[c] = k;
      const base = idx[b] * strides[b] + idx[c] * strides[c];
      for (let p = 0; p + 1 < list.length; p += 2) {
        const i0 = Math.max(0, Math.ceil(list[p] - 0.5));
        const i1 = Math.min(na - 1, Math.floor(list[p + 1] - 0.5));
        for (let i = i0; i <= i1; i++) votes[base + i * strides[a]]++;
      }
    }
  return { odd, columns };
}

/** Marks every cell touched by the surface (sampled at a quarter of a cell). */
function markSurface(tris: Float32Array, nx: number, ny: number, nz: number, solid: Uint8Array): void {
  for (let t = 0; t < tris.length; t += 9) {
    const ex = Math.hypot(tris[t + 3] - tris[t], tris[t + 4] - tris[t + 1], tris[t + 5] - tris[t + 2]);
    const ey = Math.hypot(tris[t + 6] - tris[t], tris[t + 7] - tris[t + 1], tris[t + 8] - tris[t + 2]);
    const steps = Math.max(1, Math.ceil(Math.max(ex, ey) * 4));
    for (let i = 0; i <= steps; i++)
      for (let j = 0; j <= steps - i; j++) {
        const u = i / steps, v = j / steps;
        const x = Math.floor(tris[t] + u * (tris[t + 3] - tris[t]) + v * (tris[t + 6] - tris[t]));
        const y = Math.floor(tris[t + 1] + u * (tris[t + 4] - tris[t + 1]) + v * (tris[t + 7] - tris[t + 1]));
        const z = Math.floor(tris[t + 2] + u * (tris[t + 5] - tris[t + 2]) + v * (tris[t + 8] - tris[t + 2]));
        if (x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz) solid[x + nx * (y + ny * z)] = 1;
      }
  }
}
