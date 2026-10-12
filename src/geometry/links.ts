import { C } from '../lbm/lattice';
import { FLAG } from './domain';

/**
 * Wall distances for interpolated bounce-back (Bouzidi, Firdaouss & Lallemand 2001).
 *
 * For every fluid cell next to the body and every lattice link that crosses the surface, the
 * exact mesh is ray-cast to find q in (0, 1]: the fraction of the link, measured from the fluid
 * cell centre, at which the wall lies. The solver then places the wall at its true position
 * instead of halfway between cell centres (staircase), which makes it second-order accurate.
 *
 * Storage: the boundary cell index (+1) is written in the upper bits of its flag word
 * (flag = type | index << 4), and the 18 q values are packed as bytes in 5 u32 per cell.
 */

export const LINK_WORDS = 5;
export const FLAG_MASK = 15;

export interface WallLinks {
  q: Uint32Array<ArrayBuffer>;
  cells: number;
  links: number;
  /** Links where the ray found no surface (inconsistent voxelization), q = 0.5 used. */
  misses: number;
}

export function computeWallLinks(tris: Float32Array, nx: number, ny: number, nz: number, flags: Uint32Array): WallLinks {
  const grid = binTriangles(tris, nx, ny, nz);
  const nxy = nx * ny;
  const qs: number[] = [];
  let cells = 0, links = 0, misses = 0;
  const bytes = new Uint8Array(LINK_WORDS * 4);

  for (let z = 1; z < nz - 1; z++)
    for (let y = 1; y < ny - 1; y++)
      for (let x = 1; x < nx - 1; x++) {
        const n = x + nx * (y + ny * z);
        if ((flags[n] & FLAG_MASK) !== FLAG.FLUID) continue;
        let any = false;
        bytes.fill(0);
        for (let i = 1; i < 19; i++) {
          const c = C[i];
          const s = n - c[0] - c[1] * nx - c[2] * nxy;
          if ((flags[s] & FLAG_MASK) !== FLAG.SOLID) continue;
          const px = x + 0.5, py = y + 0.5, pz = z + 0.5;
          let t = rayCast(tris, grid, nx, ny, px, py, pz, -c[0], -c[1], -c[2]);
          if (t < 0) {
            t = 0.5;
            misses++;
          }
          bytes[i - 1] = Math.max(1, Math.min(255, Math.round(t * 255)));
          any = true;
          links++;
        }
        if (!any) continue;
        cells++;
        flags[n] = (flags[n] & FLAG_MASK) | (cells << 4);
        const words = new Uint32Array(bytes.buffer);
        for (let k = 0; k < LINK_WORDS; k++) qs.push(words[k]);
      }
  return { q: new Uint32Array(qs.length ? qs : [0, 0, 0, 0, 0]), cells, links, misses };
}

interface TriGrid {
  start: Uint32Array;
  ids: Uint32Array;
}

/** Compressed list of the triangles overlapping each lattice cell (by bounding box). */
function binTriangles(tris: Float32Array, nx: number, ny: number, nz: number): TriGrid {
  const N = nx * ny * nz;
  const count = new Uint32Array(N + 1);
  const range = (t: number, cb: (cell: number) => void) => {
    const lo = [0, 1, 2].map((a) => Math.max(0, Math.floor(Math.min(tris[t + a], tris[t + 3 + a], tris[t + 6 + a]) - 1e-4)));
    const hi = [0, 1, 2].map((a, k) => Math.min([nx, ny, nz][k] - 1, Math.floor(Math.max(tris[t + a], tris[t + 3 + a], tris[t + 6 + a]) + 1e-4)));
    for (let z = lo[2]; z <= hi[2]; z++)
      for (let y = lo[1]; y <= hi[1]; y++)
        for (let x = lo[0]; x <= hi[0]; x++) cb(x + nx * (y + ny * z));
  };
  for (let t = 0; t < tris.length; t += 9) range(t, (c) => count[c + 1]++);
  for (let i = 0; i < N; i++) count[i + 1] += count[i];
  const ids = new Uint32Array(count[N]);
  const fill = count.slice(0, N);
  for (let t = 0; t < tris.length; t += 9) range(t, (c) => (ids[fill[c]++] = t));
  return { start: count, ids };
}

/** Nearest hit parameter t in (0, 1] along p + t d, or -1. Segments span at most 2 cells per axis. */
function rayCast(tris: Float32Array, g: TriGrid, nx: number, ny: number, px: number, py: number, pz: number, dx: number, dy: number, dz: number): number {
  let best = Infinity;
  const x0 = Math.floor(Math.min(px, px + dx)), x1 = Math.floor(Math.max(px, px + dx));
  const y0 = Math.floor(Math.min(py, py + dy)), y1 = Math.floor(Math.max(py, py + dy));
  const z0 = Math.floor(Math.min(pz, pz + dz)), z1 = Math.floor(Math.max(pz, pz + dz));
  for (let z = z0; z <= z1; z++)
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        const c = x + nx * (y + ny * z);
        for (let k = g.start[c]; k < g.start[c + 1]; k++) {
          const t = intersect(tris, g.ids[k], px, py, pz, dx, dy, dz);
          if (t > 1e-6 && t <= 1 + 1e-6 && t < best) best = t;
        }
      }
  return best === Infinity ? -1 : Math.min(best, 1);
}

/** Möller–Trumbore ray / triangle intersection (two-sided). */
function intersect(p: Float32Array, t: number, ox: number, oy: number, oz: number, dx: number, dy: number, dz: number): number {
  const e1x = p[t + 3] - p[t], e1y = p[t + 4] - p[t + 1], e1z = p[t + 5] - p[t + 2];
  const e2x = p[t + 6] - p[t], e2y = p[t + 7] - p[t + 1], e2z = p[t + 8] - p[t + 2];
  const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
  const a = e1x * hx + e1y * hy + e1z * hz;
  if (Math.abs(a) < 1e-12) return -1;
  const f = 1 / a;
  const sx = ox - p[t], sy = oy - p[t + 1], sz = oz - p[t + 2];
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < -1e-7 || u > 1 + 1e-7) return -1;
  const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
  const v = f * (dx * qx + dy * qy + dz * qz);
  if (v < -1e-7 || u + v > 1 + 1e-7) return -1;
  return f * (e2x * qx + e2y * qy + e2z * qz);
}
