import { C, OPP, Q, W } from './lattice';
import { FLAG } from '../geometry/domain';
import { LINK_WORDS } from '../geometry/links';

/**
 * WGSL for the lattice Boltzmann solver. Loops over the 19 directions are unrolled here in
 * TypeScript so that every distribution lives in a register and every constant is a literal.
 *
 * Storage convention ("DDF shifting"): buffers hold f_i - w_i, which keeps the small deviations
 * from rest in the significant bits of the float32 instead of losing them next to w_i.
 */

export const WG = 256;

const lit = (v: number): string => (Number.isInteger(v) ? v.toFixed(1) : v.toPrecision(10));
const vec = (c: number[]): string => `vec3f(${c.map(lit).join(', ')})`;

/** Expression for c . v where v is a vec3f expression. */
function cdot(c: number[], v: string): string {
  const terms: string[] = [];
  ['x', 'y', 'z'].forEach((a, k) => {
    if (c[k] === 1) terms.push(`+ ${v}.${a}`);
    else if (c[k] === -1) terms.push(`- ${v}.${a}`);
  });
  if (!terms.length) return '0.0';
  return `(${terms.join(' ').replace(/^\+ /, '')})`;
}

/** Signed sum over directions: sum_i coef(c_i) * f_i. */
function moment(coef: (c: number[]) => number): string {
  const terms: string[] = [];
  for (let i = 0; i < Q; i++) {
    const k = coef(C[i]);
    if (k === 1) terms.push(`+ f${i}`);
    else if (k === -1) terms.push(`- f${i}`);
    else if (k !== 0) terms.push(`+ ${lit(k)} * f${i}`);
  }
  return terms.join(' ').replace(/^\+ /, '') || '0.0';
}

/** Shifted equilibrium f_i^eq - w_i for (drho, rho, u, uu) in scope. */
function feqShift(i: number): string {
  const w = lit(W[i]);
  if (i === 0) return `${w} * (drho - 1.5 * rho * uu)`;
  const cu = cdot(C[i], 'u');
  return `${w} * (drho + rho * (3.0 * ${cu} + 4.5 * ${cu} * ${cu} - 1.5 * uu))`;
}

function storeEquilibrium(target: string): string {
  return Array.from({ length: Q }, (_, i) => `${target}[${i}u * N + n] = ${feqShift(i)};`).join('\n      ');
}

export const PARAMS_WGSL = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  wgx: u32, model: u32, _p1: u32, _p2: u32,  // model: 0 regularized, 1 BGK
  tau: f32, smag: f32, turbK: f32, _p3: f32,
  uin: vec4f,    // xyz: free-stream velocity, w: inlet turbulence intensity
  wall: vec4f,   // xyz: velocity of the ground wall
  pert: vec4f,   // xyz: centre of the mouse forcing, w: radius
  pertF: vec4f,  // xyz: force density, w: 1 when active
}`;

/** Seeded random-Fourier-modes generator for synthetic inflow turbulence (Kraichnan). */
function inflowFunction(): string {
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const M = 12;
  const lines: string[] = [];
  for (let k = 0; k < M; k++) {
    const th = Math.acos(2 * rnd() - 1), ph = 2 * Math.PI * rnd();
    const kd = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    const mag = Math.pow(4, rnd() - 0.5); // wavenumbers spread over two octaves
    const r = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
    // amplitude perpendicular to the wave vector: each mode is divergence-free
    let sig = [kd[1] * r[2] - kd[2] * r[1], kd[2] * r[0] - kd[0] * r[2], kd[0] * r[1] - kd[1] * r[0]];
    const l = Math.hypot(sig[0], sig[1], sig[2]) || 1;
    sig = sig.map((v) => v / l);
    const kv = kd.map((v) => v * mag);
    lines.push(
      `d += ${vec(sig)} * cos(P.turbK * (${lit(kv[0])} * xt + ${lit(kv[1])} * y + ${lit(kv[2])} * z) + ${lit(2 * Math.PI * rnd())});`,
    );
  }
  return /* wgsl */ `
fn inflow(y: f32, z: f32, t: f32) -> vec3f {
  let u0 = P.uin.xyz;
  if (P.uin.w <= 0.0) { return u0; }
  let U = length(u0);
  let xt = -U * t; // frozen turbulence convected through the inlet plane
  var d = vec3f(0.0);
  ${lines.join('\n  ')}
  return u0 + d * (P.uin.w * U * ${lit(Math.sqrt(6 / M))});
}`;
}

export function stepShader(): string {
  const stream: string[] = [];
  for (let i = 0; i < Q; i++) {
    if (i === 0) {
      stream.push('let f0 = fA[n];');
      continue;
    }
    const c = C[i];
    const o = OPP[i];
    const off: string[] = [];
    if (c[0]) off.push(`${c[0] > 0 ? '-' : '+'} 1`);
    if (c[1]) off.push(`${c[1] > 0 ? '-' : '+'} inx`);
    if (c[2]) off.push(`${c[2] > 0 ? '-' : '+'} inxy`);
    const off2 = off.map((t) => (t[0] === '-' ? '+' : '-') + t.slice(1));
    const pull = `{
        let s = u32(i32(n) ${off.join(' ')});
        let fs = flags[s] & 15u;
        if (fs == ${FLAG.SOLID}u) {
          let fo = fA[${o}u * N + n];
          var fb = fo;
          if (bi > 0u) {
            // interpolated bounce-back (Bouzidi): q = wall distance along the link
            let q = f32((qlinks[(bi - 1u) * ${LINK_WORDS}u + ${Math.floor((i - 1) / 4)}u] >> ${8 * ((i - 1) % 4)}u) & 255u) / 255.0;
            if (q < 0.5) {
              let n2 = u32(i32(n) ${off2.join(' ')});
              let fl2 = flags[n2] & 15u;
              var f2 = fo;
              if (fl2 == ${FLAG.FLUID}u || fl2 == ${FLAG.FREESTREAM}u || fl2 == ${FLAG.OUTLET}u) { f2 = fA[${o}u * N + n2]; }
              fb = 2.0 * q * fo + (1.0 - 2.0 * q) * f2;
            } else {
              let h = 0.5 / q;
              fb = h * fo + (1.0 - h) * fA[${i}u * N + n];
            }
          }
          f${i} = fb;
          force -= ${vec(c)} * (fo + fb + ${lit(2 * W[i])});
        } else if (fs == ${FLAG.WALL}u) {
          f${i} = fA[${o}u * N + n] + ${lit(6 * W[i])} * ${cdot(c, 'P.wall')};
        } else {
          f${i} = fA[${i}u * N + s];
        }
      }`;
    stream.push(`var f${i}: f32;`);
    // At the outlet, populations coming from outside the domain are extrapolated (zero gradient).
    stream.push(c[0] === -1 ? `if (outlet) { f${i} = fA[${i}u * N + n]; } else ${pull}` : pull);
  }

  const collide: string[] = [];
  for (let i = 0; i < Q; i++) {
    const c = C[i];
    const w = W[i];
    const qp: string[] = [];
    const sq = (k: number, name: string) => c[k] !== 0 && qp.push(`P${name}`);
    sq(0, 'xx');
    sq(1, 'yy');
    sq(2, 'zz');
    const cross = (a: number, b: number, name: string) => {
      const s = c[a] * c[b];
      if (s) qp.push(`${s > 0 ? '' : '-'}2.0 * P${name}`);
    };
    cross(0, 1, 'xy');
    cross(0, 2, 'xz');
    cross(1, 2, 'yz');
    const qpExpr = qp.length ? `${qp.join(' + ').replace(/\+ -/g, '- ')} - tr` : '-tr';
    const cu = i === 0 ? '0.0' : cdot(c, 'u');
    const cF = i === 0 ? '0.0' : cdot(c, 'F');
    const eq = feqShift(i);
    const neq = `${lit(4.5 * w)} * (${qpExpr})`;
    const src = i === 0 ? `${lit(w)} * (-3.0 * uF)` : `${lit(w)} * (3.0 * (${cF} - uF) + 9.0 * ${cu} * ${cF})`;
    collide.push(`{ let fe = ${eq}; fB[${i}u * N + n] = fe + om1 * select(${neq}, f${i} - fe, bgk) + fo * ${src}; }`);
  }

  return /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fA: array<f32>;
@group(0) @binding(2) var<storage, read_write> fB: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> mac: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> partial: array<vec4f>;
@group(0) @binding(6) var<storage, read> counter: array<u32>;
@group(0) @binding(7) var<storage, read> qlinks: array<u32>;

var<workgroup> wsum: array<vec3f, ${WG}>;

${inflowFunction()}

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let wgi = wid.x + wid.y * P.wgx;
  let n = wgi * ${WG}u + li;
  var force = vec3f(0.0);
  if (n < P.n) {
    let N = P.n;
    let flagRaw = flags[n];
    let flag = flagRaw & 15u;
    let x = n % P.nx;
    let y = (n / P.nx) % P.ny;
    let z = n / (P.nx * P.ny);
    if (flag == ${FLAG.FREESTREAM}u) {
      // Open boundaries:
      //  - inlet (x = 0): equilibrium at the imposed velocity (plus synthetic turbulence) and the
      //    reference density; it also absorbs the acoustic waves that reach it. (Floating-pressure
      //    and total-pressure inlets were tried: both reflect waves and oscillate.)
      //  - side faces: far-field pressure (rho = 1) with the velocity of the adjacent interior cell
      //    (previous step), so the flow can be displaced outward around the body instead of being
      //    confined as in a closed tunnel.
      var u = P.uin.xyz;
      let drho = 0.0;
      if (x == 0u) {
        u = inflow(f32(y) + 0.5, f32(z) + 0.5, f32(counter[0]));
      } else {
        let m = clamp(x, 1u, P.nx - 2u) + P.nx * (clamp(y, 1u, P.ny - 2u) + P.ny * clamp(z, 1u, P.nz - 2u));
        let fm = flags[m] & 15u;
        if (fm != ${FLAG.SOLID}u && fm != ${FLAG.WALL}u) {
          ${Array.from({ length: Q }, (_, i) => `let g${i} = fA[${i}u * N + m];`).join('\n          ')}
          let gs = ${moment(() => 1).replace(/f(\d+)/g, 'g$1')};
          u = vec3f(${[0, 1, 2].map((a) => moment((c) => c[a]).replace(/f(\d+)/g, 'g$1')).join(', ')}) / (1.0 + gs);
          let um = length(u);
          if (um > 0.4) { u = u * (0.4 / um); }
        }
      }
      let rho = 1.0 + drho;
      let uu = dot(u, u);
      ${storeEquilibrium('fB')}
      mac[n] = vec4f(rho, u);
    } else if (flag == ${FLAG.FLUID}u || flag == ${FLAG.OUTLET}u) {
      let outlet = flag == ${FLAG.OUTLET}u;
      let bi = flagRaw >> 4u;
      let inx = i32(P.nx);
      let inxy = i32(P.nx * P.ny);
      ${stream.join('\n      ')}

      let sum = ${moment(() => 1)};
      let jx = ${moment((c) => c[0])};
      let jy = ${moment((c) => c[1])};
      let jz = ${moment((c) => c[2])};

      var F = vec3f(0.0);
      if (P.pertF.w > 0.0) {
        let d = (vec3f(f32(x), f32(y), f32(z)) + 0.5 - P.pert.xyz) / P.pert.w;
        let r2 = dot(d, d);
        if (r2 < 9.0) { F = P.pertF.xyz * exp(-r2); }
      }

      if (outlet) {
        // Pressure outlet: rho = 1, velocity extrapolated from the interior.
        var u = vec3f(jx, jy, jz) / (1.0 + sum);
        let um = length(u);
        if (um > 0.4) { u = u * (0.4 / um); }
        let rho = 1.0;
        let drho = 0.0;
        let uu = dot(u, u);
        ${storeEquilibrium('fB')}
        mac[n] = vec4f(rho, u);
      } else {
        let drho = sum;
        let rho = 1.0 + drho;
        var u = (vec3f(jx, jy, jz) + 0.5 * F) / rho;
        let um = length(u);
        if (um > 0.4) { u = u * (0.4 / um); }
        let uu = dot(u, u);

        // Non-equilibrium momentum flux Pi^neq = sum c c (f - feq)
        let r3 = drho / 3.0;
        let Pxx = (${moment((c) => c[0] * c[0])}) - rho * u.x * u.x - r3;
        let Pyy = (${moment((c) => c[1] * c[1])}) - rho * u.y * u.y - r3;
        let Pzz = (${moment((c) => c[2] * c[2])}) - rho * u.z * u.z - r3;
        let Pxy = (${moment((c) => c[0] * c[1])}) - rho * u.x * u.y;
        let Pxz = (${moment((c) => c[0] * c[2])}) - rho * u.x * u.z;
        let Pyz = (${moment((c) => c[1] * c[2])}) - rho * u.y * u.z;
        let QQ = Pxx * Pxx + Pyy * Pyy + Pzz * Pzz + 2.0 * (Pxy * Pxy + Pxz * Pxz + Pyz * Pyz);

        // Smagorinsky LES: local relaxation time from the resolved strain (Hou et al. 1996)
        let tau0 = P.tau;
        let tauE = 0.5 * (tau0 + sqrt(tau0 * tau0 + ${lit(18 * Math.SQRT2)} * P.smag * sqrt(QQ) / rho));
        let om = 1.0 / tauE;
        let om1 = 1.0 - om;
        let fo = 1.0 - 0.5 * om;
        let tr = (Pxx + Pyy + Pzz) / 3.0;
        let uF = dot(u, F);
        let bgk = P.model == 1u;

        // Regularized collision: f = feq + (1 - omega) * fneq(Pi^neq) + Guo forcing
        ${collide.join('\n        ')}
        mac[n] = vec4f(rho, u);
      }
    }
  }

  wsum[li] = force;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) { wsum[li] = wsum[li] + wsum[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { partial[wgi] = vec4f(wsum[0], 0.0); }
}`;
}

export function initShader(): string {
  return /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> f: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> mac: array<vec4f>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let n = (wid.x + wid.y * P.wgx) * ${WG}u + li;
  if (n >= P.n) { return; }
  let N = P.n;
  let fl = flags[n] & 15u;
  var u = P.uin.xyz;
  if (fl == ${FLAG.SOLID}u || fl == ${FLAG.WALL}u) { u = vec3f(0.0); }
  let rho = 1.0;
  let drho = 0.0;
  let uu = dot(u, u);
  ${storeEquilibrium('f')}
  mac[n] = vec4f(rho, u);
}`;
}

export function reduceShader(): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> partial: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> counter: array<u32>;
@group(0) @binding(3) var<uniform> R: vec4u;

var<workgroup> ws: array<vec4f, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32) {
  var s = vec4f(0.0);
  for (var i = li; i < R.x; i = i + ${WG}u) { s = s + partial[i]; }
  ws[li] = s;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k = k >> 1u) {
    if (li < k) { ws[li] = ws[li] + ws[li + k]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    accum[0] = accum[0] + ws[0];
    accum[1] = accum[1] + vec4f(1.0, 0.0, 0.0, 0.0);
    counter[0] = counter[0] + 1u;
  }
}`;
}

const FIELD_PARAMS = /* wgsl */ `
struct FieldParams {
  nx: u32, ny: u32, nz: u32, n: u32,
  wgx: u32, sliceField: u32, volField: u32, _p: u32,
  U: f32, lref: f32, _a: f32, _b: f32,
  wall: vec4f,
}`;

/** Gives solid cells the mean density of their fluid neighbours (for smooth sampling at walls). */
export function fillShader(): string {
  return /* wgsl */ `
${FIELD_PARAMS}
@group(0) @binding(0) var<uniform> F: FieldParams;
@group(0) @binding(1) var<storage, read_write> mac: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;

fn isFluid(fl: u32) -> bool { return fl == ${FLAG.FLUID}u || fl == ${FLAG.FREESTREAM}u || fl == ${FLAG.OUTLET}u; }

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let n = (wid.x + wid.y * F.wgx) * ${WG}u + li;
  if (n >= F.n) { return; }
  let fl = flags[n] & 15u;
  if (isFluid(fl)) { return; }
  let p = vec3i(i32(n % F.nx), i32((n / F.nx) % F.ny), i32(n / (F.nx * F.ny)));
  let d = vec3i(i32(F.nx), i32(F.ny), i32(F.nz));
  var s = 0.0;
  var c = 0.0;
  for (var k = 0; k < 6; k = k + 1) {
    var o = vec3i(0);
    o[k / 2] = select(-1, 1, (k & 1) == 1);
    let q = p + o;
    if (all(q >= vec3i(0)) && all(q < d)) {
      let m = u32(q.x + d.x * (q.y + d.y * q.z));
      if (isFluid(flags[m] & 15u)) { s = s + mac[m].x; c = c + 1.0; }
    }
  }
  var u = vec3f(0.0);
  if (fl == ${FLAG.WALL}u) { u = F.wall.xyz; }
  mac[n] = vec4f(select(1.0, s / max(c, 1.0), c > 0.0), u);
}`;
}

/** Field ids, shared with the UI. */
export const SLICE_FIELDS = [
  { id: 0, label: 'Vitesse |u| / U∞', range: [0, 1.5], diverging: false },
  { id: 1, label: 'Vitesse axiale ux / U∞', range: [-0.4, 1.4], diverging: false },
  { id: 2, label: 'Vitesse verticale uz / U∞', range: [-0.6, 0.6], diverging: true },
  { id: 3, label: 'Vitesse latérale uy / U∞', range: [-0.6, 0.6], diverging: true },
  { id: 4, label: 'Pression Cp', range: [-1.5, 1], diverging: true },
  { id: 5, label: 'Pression totale Cpt', range: [0, 1.1], diverging: false },
  { id: 6, label: 'Vorticité |ω|·L/U∞', range: [0, 30], diverging: false },
  { id: 7, label: 'Vorticité axiale ωx·L/U∞', range: [-20, 20], diverging: true },
  { id: 8, label: 'Critère Q·(L/U∞)²', range: [-200, 200], diverging: true },
] as const;

export const BRICK = 8;

export function fieldsShader(): string {
  return /* wgsl */ `
${FIELD_PARAMS}
@group(0) @binding(0) var<uniform> F: FieldParams;
@group(0) @binding(1) var<storage, read> mac: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> scal: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> bricks: array<atomic<u32>>;

fn at(p: vec3i) -> vec3f {
  let d = vec3i(i32(F.nx), i32(F.ny), i32(F.nz));
  let q = clamp(p, vec3i(0), d - 1);
  return mac[u32(q.x + d.x * (q.y + d.y * q.z))].yzw;
}

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let n = (wid.x + wid.y * F.wgx) * ${WG}u + li;
  if (n >= F.n) { return; }
  let p = vec3i(i32(n % F.nx), i32((n / F.nx) % F.ny), i32(n / (F.nx * F.ny)));
  let m = mac[n];
  let u = m.yzw;
  let U = F.U;
  // velocity gradient by central differences: gx = d(u,v,w)/dx, ...
  let gx = (at(p + vec3i(1, 0, 0)) - at(p - vec3i(1, 0, 0))) * 0.5;
  let gy = (at(p + vec3i(0, 1, 0)) - at(p - vec3i(0, 1, 0))) * 0.5;
  let gz = (at(p + vec3i(0, 0, 1)) - at(p - vec3i(0, 0, 1))) * 0.5;
  let w = vec3f(gy.z - gz.y, gz.x - gx.z, gx.y - gy.x);
  let sxy = 0.5 * (gy.x + gx.y);
  let sxz = 0.5 * (gz.x + gx.z);
  let syz = 0.5 * (gz.y + gy.z);
  let S2 = gx.x * gx.x + gy.y * gy.y + gz.z * gz.z + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
  let qcrit = 0.5 * (0.5 * dot(w, w) - S2);
  let tU = F.lref / U;
  let cp = (m.x - 1.0) / (1.5 * U * U);

  var v = 0.0;
  switch F.sliceField {
    case 0u: { v = length(u) / U; }
    case 1u: { v = u.x / U; }
    case 2u: { v = u.z / U; }
    case 3u: { v = u.y / U; }
    case 4u: { v = cp; }
    case 5u: { v = cp + dot(u, u) / (U * U); }
    case 6u: { v = length(w) * tU; }
    case 7u: { v = w.x * tU; }
    default: { v = qcrit * tU * tU; }
  }

  var vol = 0.0;
  if (F.volField == 1u) { vol = qcrit * tU * tU; }
  else if (F.volField == 2u) { vol = length(w) * tU; }
  let fl = flags[n] & 15u;
  if (fl == ${FLAG.SOLID}u || fl == ${FLAG.WALL}u || fl == ${FLAG.FREESTREAM}u) { vol = 0.0; }
  scal[n] = vec2f(v, vol);

  if (vol > 0.0) {
    // max per ${BRICK}^3 brick, dilated by one cell, for empty-space skipping in the ray marcher
    let bd = (vec3i(i32(F.nx), i32(F.ny), i32(F.nz)) + ${BRICK - 1}) / ${BRICK};
    let lo = max((p - 1) / ${BRICK}, vec3i(0));
    let hi = min((p + 1) / ${BRICK}, bd - 1);
    let bits = bitcast<u32>(vol);
    for (var bz = lo.z; bz <= hi.z; bz = bz + 1) {
      for (var by = lo.y; by <= hi.y; by = by + 1) {
        for (var bx = lo.x; bx <= hi.x; bx = bx + 1) {
          atomicMax(&bricks[u32(bx + bd.x * (by + bd.y * bz))], bits);
        }
      }
    }
  }
}`;
}
