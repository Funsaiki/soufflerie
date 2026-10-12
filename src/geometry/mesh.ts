import type { V3 } from '../render/math';

/** Triangle soup: 9 floats (3 vertices) per triangle. */
export interface Mesh {
  name: string;
  positions: Float32Array;
}

export interface Bounds {
  min: V3;
  max: V3;
}

export function bounds(p: Float32Array): Bounds {
  const min: V3 = [Infinity, Infinity, Infinity];
  const max: V3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = p[i + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  return { min, max };
}

export function triangleCount(m: Mesh): number {
  return m.positions.length / 9;
}

/** Rotation matrix (row-major 3x3) R = Rz(yaw) * Ry(pitch) * Rx(roll), angles in degrees. */
export function rotationMatrix(yawDeg: number, pitchDeg: number, rollDeg: number): number[] {
  const [y, p, r] = [yawDeg, pitchDeg, rollDeg].map((d) => (d * Math.PI) / 180);
  const cy = Math.cos(y), sy = Math.sin(y);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cr = Math.cos(r), sr = Math.sin(r);
  // Ry uses the convention x' = x cos + z sin, z' = -x sin + z cos so that a positive pitch
  // raises the leading edge (the -x end, facing the incoming flow): positive angle of attack.
  return [
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr,
  ];
}

export function applyMatrix3(p: Float32Array, m: number[]): Float32Array {
  const out = new Float32Array(p.length);
  for (let i = 0; i < p.length; i += 3) {
    const x = p[i], y = p[i + 1], z = p[i + 2];
    out[i] = m[0] * x + m[1] * y + m[2] * z;
    out[i + 1] = m[3] * x + m[4] * y + m[5] * z;
    out[i + 2] = m[6] * x + m[7] * y + m[8] * z;
  }
  return out;
}

/** Converts a Y-up model (glTF/OBJ convention) to the Z-up convention used here. */
export function yUpToZUp(p: Float32Array): Float32Array {
  const out = new Float32Array(p.length);
  for (let i = 0; i < p.length; i += 3) {
    out[i] = p[i + 2];
    out[i + 1] = p[i];
    out[i + 2] = p[i + 1];
  }
  return out;
}

/** Interleaved position + flat normal (6 floats per vertex) for rendering. */
export function withFlatNormals(p: Float32Array): Float32Array<ArrayBuffer> {
  const out = new Float32Array((p.length / 3) * 6);
  for (let t = 0; t < p.length; t += 9) {
    const ax = p[t + 3] - p[t], ay = p[t + 4] - p[t + 1], az = p[t + 5] - p[t + 2];
    const bx = p[t + 6] - p[t], by = p[t + 7] - p[t + 1], bz = p[t + 8] - p[t + 2];
    let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    for (let v = 0; v < 3; v++) {
      const o = ((t / 3) + v) * 6;
      out[o] = p[t + v * 3];
      out[o + 1] = p[t + v * 3 + 1];
      out[o + 2] = p[t + v * 3 + 2];
      out[o + 3] = nx;
      out[o + 4] = ny;
      out[o + 5] = nz;
    }
  }
  return out;
}
