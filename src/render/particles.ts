import { FLAG } from '../geometry/domain';

/** Smoke tracers advected through the velocity field (RK2), emitted from a rake of nozzles. */

export type RakeMode = 'horizontal' | 'vertical' | 'grid' | 'cloud';

export interface RakeSettings {
  mode: RakeMode;
  /** Rake centre (lattice units) and half-extent. */
  y: number;
  z: number;
  halfY: number;
  halfZ: number;
  nozzles: number;
}

const SHADER = /* wgsl */ `
struct PP {
  dims: vec4f,   // nx, ny, nz, dt (lattice steps since last update)
  rake: vec4f,   // y0, y1, z0, z1
  rake2: vec4f,  // x0, nozzles along y, nozzles along z, lifetime
  seed: vec4u,   // frame, count, cloud (0/1), unused
}
@group(0) @binding(0) var<uniform> P: PP;
@group(0) @binding(1) var<storage, read_write> parts: array<vec4f>;
@group(0) @binding(2) var<storage, read> mac: array<vec4f>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rnd(s: ptr<function, u32>) -> f32 {
  *s = pcg(*s);
  return f32(*s) / 4294967296.0;
}

fn cellIndex(i: vec3i) -> u32 {
  let d = vec3i(P.dims.xyz);
  let c = clamp(i, vec3i(0), d - 1);
  return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn sampleU(p: vec3f) -> vec3f {
  let q = p - 0.5;
  let i0 = vec3i(floor(q));
  let t = q - floor(q);
  var r = vec3f(0.0);
  for (var c = 0u; c < 8u; c = c + 1u) {
    let o = vec3i(i32(c & 1u), i32((c >> 1u) & 1u), i32((c >> 2u) & 1u));
    let w = select(1.0 - t, t, o == vec3i(1));
    r = r + w.x * w.y * w.z * mac[cellIndex(i0 + o)].yzw;
  }
  return r;
}

fn nozzle(s: ptr<function, u32>) -> vec2f {
  let ny = P.rake2.y;
  let nz = P.rake2.z;
  var a = rnd(s);
  var b = rnd(s);
  if (P.seed.z == 0u) {
    a = (floor(a * ny) + 0.5) / ny + (rnd(s) - 0.5) * 0.05 / ny;
    b = (floor(b * nz) + 0.5) / nz + (rnd(s) - 0.5) * 0.05 / nz;
  }
  return vec2f(mix(P.rake.x, P.rake.y, a), mix(P.rake.z, P.rake.w, b));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.seed.y) { return; }
  var s = pcg(i * 9781u + pcg(P.seed.x));
  var p = parts[i];
  let d = P.dims.xyz;
  let life = P.rake2.w * (0.6 + 0.4 * f32(pcg(i) & 1023u) / 1023.0);
  if (p.w < 0.0) {
    // first frame: spread along the tunnel so the flow is populated right away
    let yz = nozzle(&s);
    p = vec4f(mix(P.rake2.x, d.x - 2.0, rnd(&s)), yz, rnd(&s) * life);
  } else {
    let dt = P.dims.w;
    let u1 = sampleU(p.xyz);
    let u2 = sampleU(p.xyz + 0.5 * dt * u1);
    p = vec4f(p.xyz + dt * u2, p.w + dt);
    let f = flags[cellIndex(vec3i(floor(p.xyz)))] & 15u;
    let out = any(p.xyz < vec3f(1.0)) || any(p.xyz > d - 1.0);
    if (out || p.w > life || f == ${FLAG.SOLID}u || f == ${FLAG.WALL}u) {
      let yz = nozzle(&s);
      p = vec4f(P.rake2.x + rnd(&s) * max(dt * length(u1), 0.5), yz, 0.0);
    }
  }
  parts[i] = p;
}`;

export class Particles {
  readonly buffer: GPUBuffer;
  private pipe: GPUComputePipeline;
  private bg: GPUBindGroup;
  private ubo: GPUBuffer;
  private frame = 0;

  constructor(private device: GPUDevice, readonly capacity: number, macro: GPUBuffer, flags: GPUBuffer) {
    this.buffer = device.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'particles' });
    this.ubo = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: SHADER, label: 'particles' }), entryPoint: 'main' },
    });
    this.bg = device.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [this.ubo, this.buffer, macro, flags].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.reset();
  }

  reset(): void {
    this.device.queue.writeBuffer(this.buffer, 0, new Float32Array(this.capacity * 4).fill(-1));
  }

  update(enc: GPUCommandEncoder, dims: [number, number, number], dt: number, rake: RakeSettings, count: number, lifetime: number): void {
    const buf = new ArrayBuffer(64);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    let { halfY, halfZ } = rake;
    let ny = rake.nozzles, nz = rake.nozzles;
    if (rake.mode === 'horizontal') { halfZ = 0; nz = 1; }
    if (rake.mode === 'vertical') { halfY = 0; ny = 1; }
    f.set([...dims, dt, rake.y - halfY, rake.y + halfY, rake.z - halfZ, rake.z + halfZ, 2.5, ny, nz, lifetime]);
    u.set([this.frame++, count, rake.mode === 'cloud' ? 1 : 0, 0], 12);
    this.device.queue.writeBuffer(this.ubo, 0, buf);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.bg);
    pass.dispatchWorkgroups(Math.ceil(count / 256));
    pass.end();
  }

  destroy(): void {
    this.buffer.destroy();
    this.ubo.destroy();
  }
}
