import { BRICK } from '../lbm/shaders';
import { FLAG } from '../geometry/domain';

/** Uniform block shared by every render pipeline (288 bytes). */
export const VIEW_WGSL = /* wgsl */ `
struct View {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  camPos: vec4f,    // xyz eye, w particle size (px)
  dims: vec4f,      // nx, ny, nz, U (lattice free-stream speed)
  slice: vec4f,     // axis, position, diverging colormap (0/1), unused
  range: vec4f,     // slice min, slice max, volume threshold, volume max
  viewport: vec4f,  // width, height, ground (0/1), particle speed max
  pert: vec4f,      // xyz forcing centre, w radius (<= 0: hidden)
  surf: vec4f,      // surface mode (0 plain, 1 Cp), Cp min, Cp max, fog density
  iso: vec4f,       // iso colour min, iso colour max, volume mode, unused
  right: vec4f,
  up: vec4f,
}
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> mac: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> scal: array<vec2f>;
@group(0) @binding(4) var<storage, read> parts: array<vec4f>;
@group(0) @binding(5) var<storage, read> bricks: array<u32>;

fn cellIndex(i: vec3i) -> u32 {
  let d = vec3i(V.dims.xyz);
  let c = clamp(i, vec3i(0), d - 1);
  return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn corner(c: u32) -> vec3i { return vec3i(i32(c & 1u), i32((c >> 1u) & 1u), i32((c >> 2u) & 1u)); }

fn cornerWeight(o: vec3i, t: vec3f) -> f32 {
  let w = select(1.0 - t, t, o == vec3i(1));
  return w.x * w.y * w.z;
}

fn sampleMacro(p: vec3f) -> vec4f {
  let q = p - 0.5;
  let i0 = vec3i(floor(q));
  let t = q - floor(q);
  var r = vec4f(0.0);
  for (var c = 0u; c < 8u; c = c + 1u) {
    let o = corner(c);
    r = r + cornerWeight(o, t) * mac[cellIndex(i0 + o)];
  }
  return r;
}

fn sampleScal(p: vec3f) -> vec2f {
  let q = p - 0.5;
  let i0 = vec3i(floor(q));
  let t = q - floor(q);
  var r = vec2f(0.0);
  for (var c = 0u; c < 8u; c = c + 1u) {
    let o = corner(c);
    r = r + cornerWeight(o, t) * scal[cellIndex(i0 + o)];
  }
  return r;
}

fn isSolidAt(p: vec3f) -> bool {
  let f = flags[cellIndex(vec3i(floor(p)))] & 15u;
  return f == ${FLAG.SOLID}u || f == ${FLAG.WALL}u;
}

fn turbo(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  let kR = vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  let kG = vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  let kB = vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  let kR2 = vec2f(-152.94239396, 59.28637943);
  let kG2 = vec2f(4.27729857, 2.82956604);
  let kB2 = vec2f(-89.90310912, 27.34824973);
  let v4 = vec4f(1.0, t, t * t, t * t * t);
  let v2 = v4.zw * v4.z;
  return clamp(vec3f(dot(v4, kR) + dot(v2, kR2), dot(v4, kG) + dot(v2, kG2), dot(v4, kB) + dot(v2, kB2)), vec3f(0.0), vec3f(1.0));
}

fn diverging(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  let lo = vec3f(0.230, 0.299, 0.754);
  let mid = vec3f(0.865, 0.865, 0.865);
  let hi = vec3f(0.706, 0.016, 0.150);
  if (t < 0.5) { return mix(lo, mid, t * 2.0); }
  return mix(mid, hi, t * 2.0 - 2.0 * 0.5);
}

fn brickDims() -> vec3i { return (vec3i(V.dims.xyz) + ${BRICK - 1}) / ${BRICK}; }

fn brickMax(p: vec3f) -> f32 {
  let bd = brickDims();
  let b = clamp(vec3i(floor(p / ${BRICK}.0)), vec3i(0), bd - 1);
  return bitcast<f32>(bricks[u32(b.x + bd.x * (b.y + bd.y * b.z))]);
}

/** Ray parameter at which the ray leaves the brick containing p. */
fn brickExit(p: vec3f, rd: vec3f) -> f32 {
  let b0 = floor(p / ${BRICK}.0) * ${BRICK}.0;
  let b1 = b0 + ${BRICK}.0;
  let inv = 1.0 / select(rd, vec3f(1e-8), abs(rd) < vec3f(1e-8));
  let tm = (select(b0, b1, rd > vec3f(0.0)) - p) * inv;
  return max(min(tm.x, min(tm.y, tm.z)), 0.0) + 0.02;
}

fn boxHit(ro: vec3f, rd: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
  let inv = 1.0 / select(rd, vec3f(1e-8), abs(rd) < vec3f(1e-8));
  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  return vec2f(max(max(tn.x, tn.y), tn.z), min(min(tf.x, tf.y), tf.z));
}

fn cameraRay(fc: vec2f) -> array<vec3f, 2> {
  let ndc = vec2f(fc.x / V.viewport.x * 2.0 - 1.0, 1.0 - fc.y / V.viewport.y * 2.0);
  let pn = V.invViewProj * vec4f(ndc, 0.0, 1.0);
  let pf = V.invViewProj * vec4f(ndc, 1.0, 1.0);
  let ro = pn.xyz / pn.w;
  return array<vec3f, 2>(ro, normalize(pf.xyz / pf.w - ro));
}
`;

export const SCENE_WGSL = /* wgsl */ `
${VIEW_WGSL}

// ---------------------------------------------------------------- domain box

struct LineOut { @builtin(position) pos: vec4f, @location(0) col: vec4f }

@vertex fn vsBox(@builtin(vertex_index) vi: u32) -> LineOut {
  var e = array<vec2u, 12>(vec2u(0u, 1u), vec2u(2u, 3u), vec2u(4u, 5u), vec2u(6u, 7u),
                           vec2u(0u, 2u), vec2u(1u, 3u), vec2u(4u, 6u), vec2u(5u, 7u),
                           vec2u(0u, 4u), vec2u(1u, 5u), vec2u(2u, 6u), vec2u(3u, 7u));
  let c = e[vi / 2u][vi % 2u];
  let p = vec3f(f32(c & 1u), f32((c >> 1u) & 1u), f32((c >> 2u) & 1u)) * V.dims.xyz;
  var o: LineOut;
  o.pos = V.viewProj * vec4f(p, 1.0);
  o.col = vec4f(0.45, 0.5, 0.58, 1.0);
  return o;
}

@fragment fn fsLine(i: LineOut) -> @location(0) vec4f { return i.col; }

// ---------------------------------------------------------------- ground

struct GroundOut { @builtin(position) pos: vec4f, @location(0) world: vec3f }

@vertex fn vsGround(@builtin(vertex_index) vi: u32) -> GroundOut {
  var q = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let p = vec3f(q[vi] * V.dims.xy, 1.0);
  var o: GroundOut;
  o.pos = V.viewProj * vec4f(p, 1.0);
  o.world = p;
  return o;
}

@fragment fn fsGround(i: GroundOut) -> @location(0) vec4f {
  let g = abs(fract(i.world.xy / 10.0 - 0.5) - 0.5) * 10.0;
  let line = 1.0 - smoothstep(0.0, 0.15, min(g.x, g.y));
  return vec4f(mix(vec3f(0.16, 0.17, 0.2), vec3f(0.3, 0.32, 0.36), line), 1.0);
}

// ---------------------------------------------------------------- slice plane

struct SliceOut { @builtin(position) pos: vec4f, @location(0) world: vec3f }

@vertex fn vsSlice(@builtin(vertex_index) vi: u32) -> SliceOut {
  var q = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let c = q[vi];
  let d = V.dims.xyz;
  let s = V.slice.y;
  var p: vec3f;
  let axis = u32(V.slice.x);
  if (axis == 0u) { p = vec3f(s, c.x * d.y, c.y * d.z); }
  else if (axis == 1u) { p = vec3f(c.x * d.x, s, c.y * d.z); }
  else { p = vec3f(c.x * d.x, c.y * d.y, s); }
  var o: SliceOut;
  o.pos = V.viewProj * vec4f(p, 1.0);
  o.world = p;
  return o;
}

@fragment fn fsSlice(i: SliceOut) -> @location(0) vec4f {
  if (isSolidAt(i.world)) { discard; }
  let v = sampleScal(i.world).x;
  let t = (v - V.range.x) / (V.range.y - V.range.x);
  var col = select(turbo(t), diverging(t), V.slice.z > 0.5);
  // faint iso-contours
  let band = abs(fract(t * 12.0) - 0.5);
  col = col * (0.88 + 0.12 * smoothstep(0.0, 0.08, band));
  return vec4f(col, 1.0);
}

// ---------------------------------------------------------------- body

struct ObjOut { @builtin(position) pos: vec4f, @location(0) world: vec3f, @location(1) normal: vec3f }

@vertex fn vsObject(@location(0) p: vec3f, @location(1) n: vec3f) -> ObjOut {
  var o: ObjOut;
  o.pos = V.viewProj * vec4f(p, 1.0);
  o.world = p;
  o.normal = n;
  return o;
}

@fragment fn fsObject(i: ObjOut) -> @location(0) vec4f {
  let n = normalize(i.normal);
  let vd = normalize(V.camPos.xyz - i.world);
  let nf = select(-n, n, dot(n, vd) > 0.0);
  var base = vec3f(0.7, 0.72, 0.76);
  if (V.surf.x > 0.5) {
    // sample the pressure just outside the wall, on whichever side is fluid
    var p = i.world + n * 1.3;
    if (isSolidAt(p)) { p = i.world - n * 1.3; }
    let U = V.dims.w;
    let cp = (sampleMacro(p).x - 1.0) / (1.5 * U * U);
    base = diverging((cp - V.surf.y) / (V.surf.z - V.surf.y));
  }
  let l = normalize(vec3f(-0.35, -0.55, 0.75));
  let diff = 0.3 + 0.45 * max(dot(nf, l), 0.0) + 0.35 * max(dot(nf, vd), 0.0);
  let h = normalize(l + vd);
  let spec = 0.18 * pow(max(dot(nf, h), 0.0), 40.0);
  return vec4f(base * diff + spec, 1.0);
}

// ---------------------------------------------------------------- particles

struct PartOut { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) uv: vec2f }

@vertex fn vsParticle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PartOut {
  var o: PartOut;
  let p = parts[ii];
  if (p.w < 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  var q = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = q[vi];
  let clip = V.viewProj * vec4f(p.xyz, 1.0);
  o.pos = clip + vec4f(c * V.camPos.w / V.viewport.xy * clip.w, 0.0, 0.0);
  let sp = length(sampleMacro(p.xyz).yzw) / V.dims.w;
  o.col = turbo(sp / V.viewport.w);
  o.uv = c;
  return o;
}

@fragment fn fsParticle(i: PartOut) -> @location(0) vec4f {
  let r2 = dot(i.uv, i.uv);
  if (r2 > 1.0) { discard; }
  return vec4f(i.col * (1.0 - 0.35 * r2), 1.0);
}

// ---------------------------------------------------------------- full-screen helpers

struct FullOut { @builtin(position) pos: vec4f }

@vertex fn vsFull(@builtin(vertex_index) vi: u32) -> FullOut {
  var q = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: FullOut;
  o.pos = vec4f(q[vi], 0.0, 1.0);
  return o;
}

fn gradVol(p: vec3f) -> vec3f {
  let e = 0.75;
  return vec3f(
    sampleScal(p + vec3f(e, 0.0, 0.0)).y - sampleScal(p - vec3f(e, 0.0, 0.0)).y,
    sampleScal(p + vec3f(0.0, e, 0.0)).y - sampleScal(p - vec3f(0.0, e, 0.0)).y,
    sampleScal(p + vec3f(0.0, 0.0, e)).y - sampleScal(p - vec3f(0.0, 0.0, e)).y);
}

struct IsoOut { @location(0) col: vec4f, @builtin(frag_depth) depth: f32 }

/** Iso-surface of the volume field (Q-criterion or |vorticity|), coloured by velocity. */
@fragment fn fsIso(i: FullOut) -> IsoOut {
  let r = cameraRay(i.pos.xy);
  let ro = r[0];
  let rd = r[1];
  let hit = boxHit(ro, rd, vec3f(1.0), V.dims.xyz - 1.0);
  if (hit.y <= max(hit.x, 0.0)) { discard; }
  let thr = V.range.z;
  var t = max(hit.x, 0.0);
  let dt = 0.6;
  var found = false;
  for (var k = 0; k < 4096; k = k + 1) {
    if (t > hit.y) { break; }
    let p = ro + rd * t;
    if (brickMax(p) < thr) { t = t + brickExit(p, rd); continue; }
    if (sampleScal(p).y > thr) {
      var a = max(t - dt, 0.0);
      var b = t;
      for (var j = 0; j < 6; j = j + 1) {
        let m = 0.5 * (a + b);
        if (sampleScal(ro + rd * m).y > thr) { b = m; } else { a = m; }
      }
      t = b;
      found = true;
      break;
    }
    t = t + dt;
  }
  if (!found) { discard; }
  let p = ro + rd * t;
  var n = -normalize(gradVol(p) + vec3f(1e-9));
  if (dot(n, rd) > 0.0) { n = -n; }
  let sp = length(sampleMacro(p).yzw) / V.dims.w;
  let base = turbo((sp - V.iso.x) / (V.iso.y - V.iso.x));
  let l = normalize(vec3f(-0.35, -0.55, 0.75));
  let diff = 0.35 + 0.45 * max(dot(n, l), 0.0) + 0.3 * max(dot(n, -rd), 0.0);
  var o: IsoOut;
  o.col = vec4f(base * diff, 1.0);
  let clip = V.viewProj * vec4f(p, 1.0);
  o.depth = clip.z / clip.w;
  return o;
}
`;

/** Second pass (no depth attachment): translucent vorticity fog and the mouse-forcing marker. */
export const OVERLAY_WGSL = /* wgsl */ `
${VIEW_WGSL}
@group(1) @binding(0) var depthTex: texture_depth_2d;

struct FullOut { @builtin(position) pos: vec4f }

@vertex fn vsFull(@builtin(vertex_index) vi: u32) -> FullOut {
  var q = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: FullOut;
  o.pos = vec4f(q[vi], 0.0, 1.0);
  return o;
}

@fragment fn fsFog(i: FullOut) -> @location(0) vec4f {
  let r = cameraRay(i.pos.xy);
  let ro = r[0];
  let rd = r[1];
  let hit = boxHit(ro, rd, vec3f(1.0), V.dims.xyz - 1.0);
  // stop at the opaque scene
  let d = textureLoad(depthTex, vec2i(i.pos.xy), 0);
  let ndc = vec2f(i.pos.x / V.viewport.x * 2.0 - 1.0, 1.0 - i.pos.y / V.viewport.y * 2.0);
  let ps = V.invViewProj * vec4f(ndc, d, 1.0);
  let tScene = dot(ps.xyz / ps.w - ro, rd);
  let tEnd = min(hit.y, tScene);
  var t = max(hit.x, 0.0);
  if (tEnd <= t) { discard; }
  let thr = V.range.z;
  let vmax = max(V.range.w, thr + 1e-3);
  let dens = V.surf.w;
  let dt = 0.7;
  var col = vec3f(0.0);
  var alpha = 0.0;
  for (var k = 0; k < 4096; k = k + 1) {
    if (t > tEnd || alpha > 0.97) { break; }
    let p = ro + rd * t;
    if (brickMax(p) < thr) { t = t + brickExit(p, rd); continue; }
    let v = sampleScal(p).y;
    if (v > thr) {
      let x = clamp((v - thr) / (vmax - thr), 0.0, 1.0);
      let a = 1.0 - exp(-dens * x * dt);
      col = col + (1.0 - alpha) * a * turbo(0.15 + 0.85 * x);
      alpha = alpha + (1.0 - alpha) * a;
    }
    t = t + dt;
  }
  if (alpha < 0.003) { discard; }
  return vec4f(col, alpha);
}

struct PertOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex fn vsPert(@builtin(vertex_index) vi: u32) -> PertOut {
  var q = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = q[vi];
  let p = V.pert.xyz + (V.right.xyz * c.x + V.up.xyz * c.y) * V.pert.w * 1.6;
  var o: PertOut;
  o.pos = V.viewProj * vec4f(p, 1.0);
  o.uv = c;
  return o;
}

@fragment fn fsPert(i: PertOut) -> @location(0) vec4f {
  let d = length(i.uv);
  if (d > 1.0) { discard; }
  let ring = smoothstep(0.86, 0.92, d) * (1.0 - smoothstep(0.95, 1.0, d));
  let a = max(ring * 0.9, 0.10 * (1.0 - d));
  return vec4f(vec3f(1.0, 0.95, 0.75) * a, a);
}
`;
