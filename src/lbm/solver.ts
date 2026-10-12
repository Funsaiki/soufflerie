import type { V3 } from '../render/math';
import { Q } from './lattice';
import { BRICK, WG, fieldsShader, fillShader, initShader, reduceShader, stepShader } from './shaders';

export interface StepParams {
  tau: number;
  /** Smagorinsky constant squared. */
  smag: number;
  uin: V3;
  turbulence: number;
  /** Base wavenumber of the inflow turbulence (rad / cell). */
  turbK: number;
  wall: V3;
  pertPos: V3;
  pertRadius: number;
  pertForce: V3;
  pertActive: boolean;
  /** Collision model: 0 regularized (robust), 1 BGK (less dissipative). */
  model?: number;
}

export interface ForceSample {
  /** Mean force on the body over the sampled steps, lattice units. */
  force: V3;
  steps: number;
}

/** GPU lattice Boltzmann solver (D3Q19, regularized collision, Smagorinsky LES). */
export class Solver {
  readonly n: number;
  readonly wgx: number;
  readonly wgy: number;
  readonly macro: GPUBuffer;
  readonly flags: GPUBuffer;
  readonly scal: GPUBuffer;
  readonly bricks: GPUBuffer;
  readonly brickDims: V3;
  steps = 0;

  private fA: GPUBuffer;
  private fB: GPUBuffer;
  private partial: GPUBuffer;
  private accum: GPUBuffer;
  private counter: GPUBuffer;
  private qlinks: GPUBuffer;
  private staging: GPUBuffer;
  private params: GPUBuffer;
  private reduceParams: GPUBuffer;
  private fieldParams: GPUBuffer;
  private stepPipe: GPUComputePipeline;
  private reducePipe: GPUComputePipeline;
  private initPipe: GPUComputePipeline;
  private fillPipe: GPUComputePipeline;
  private fieldsPipe: GPUComputePipeline;
  private bgAB: GPUBindGroup;
  private bgBA: GPUBindGroup;
  private bgReduce: GPUBindGroup;
  private bgInit: GPUBindGroup;
  private bgFill: GPUBindGroup;
  private bgFields: GPUBindGroup;
  private parity = 0;
  private stagingState: 'idle' | 'copied' | 'mapping' = 'idle';
  private paramData = new ArrayBuffer(112);
  private lastParams: StepParams | null = null;

  constructor(private device: GPUDevice, readonly nx: number, readonly ny: number, readonly nz: number, flags: Uint32Array<ArrayBuffer>, qlinks: Uint32Array<ArrayBuffer>) {
    this.n = nx * ny * nz;
    const groups = Math.ceil(this.n / WG);
    this.wgx = Math.min(groups, 32768);
    this.wgy = Math.ceil(groups / this.wgx);
    const S = GPUBufferUsage.STORAGE;
    const fBytes = Q * 4 * this.n;
    const mk = (size: number, usage: number, label: string) => device.createBuffer({ size, usage, label });
    this.fA = mk(fBytes, S, 'fA');
    this.fB = mk(fBytes, S, 'fB');
    this.flags = mk(4 * this.n, S | GPUBufferUsage.COPY_DST, 'flags');
    device.queue.writeBuffer(this.flags, 0, flags);
    this.qlinks = mk(Math.max(qlinks.byteLength, 16), S | GPUBufferUsage.COPY_DST, 'qlinks');
    device.queue.writeBuffer(this.qlinks, 0, qlinks);
    this.macro = mk(16 * this.n, S | GPUBufferUsage.COPY_SRC, 'macro');
    this.scal = mk(8 * this.n, S, 'scalars');
    this.brickDims = [Math.ceil(nx / BRICK), Math.ceil(ny / BRICK), Math.ceil(nz / BRICK)];
    this.bricks = mk(4 * Math.max(1, this.brickDims[0] * this.brickDims[1] * this.brickDims[2]), S | GPUBufferUsage.COPY_DST, 'bricks');
    this.partial = mk(16 * this.wgx * this.wgy, S, 'partial');
    this.accum = mk(32, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'accum');
    this.counter = mk(16, S | GPUBufferUsage.COPY_DST, 'counter');
    this.staging = mk(32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'staging');
    this.params = mk(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'params');
    this.reduceParams = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'reduceParams');
    this.fieldParams = mk(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'fieldParams');
    device.queue.writeBuffer(this.reduceParams, 0, new Uint32Array([this.wgx * this.wgy, 0, 0, 0]));

    const pipe = (code: string, label: string) =>
      device.createComputePipeline({
        label,
        layout: 'auto',
        compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' },
      });
    this.stepPipe = pipe(stepShader(), 'lbm-step');
    this.reducePipe = pipe(reduceShader(), 'lbm-reduce');
    this.initPipe = pipe(initShader(), 'lbm-init');
    this.fillPipe = pipe(fillShader(), 'fill');
    this.fieldsPipe = pipe(fieldsShader(), 'fields');

    const bg = (p: GPUComputePipeline, buffers: GPUBuffer[]) =>
      device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    this.bgAB = bg(this.stepPipe, [this.params, this.fA, this.fB, this.flags, this.macro, this.partial, this.counter, this.qlinks]);
    this.bgBA = bg(this.stepPipe, [this.params, this.fB, this.fA, this.flags, this.macro, this.partial, this.counter, this.qlinks]);
    this.bgReduce = bg(this.reducePipe, [this.partial, this.accum, this.counter, this.reduceParams]);
    this.bgInit = bg(this.initPipe, [this.params, this.fA, this.flags, this.macro]);
    this.bgFill = bg(this.fillPipe, [this.fieldParams, this.macro, this.flags]);
    this.bgFields = bg(this.fieldsPipe, [this.fieldParams, this.macro, this.flags, this.scal, this.bricks]);
  }

  static bytesFor(cells: number): number {
    return cells * (2 * Q * 4 + 4 + 16 + 8);
  }

  setParams(p: StepParams): void {
    this.lastParams = p;
    const u = new Uint32Array(this.paramData);
    const f = new Float32Array(this.paramData);
    u.set([this.nx, this.ny, this.nz, this.n, this.wgx, p.model ?? 0, 0, 0]);
    f.set([p.tau, p.smag, p.turbK, 0], 8);
    f.set([...p.uin, p.turbulence], 12);
    f.set([...p.wall, 0], 16);
    f.set([...p.pertPos, Math.max(p.pertRadius, 0.5)], 20);
    f.set([...p.pertForce, p.pertActive ? 1 : 0], 24);
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  /** Resets the flow to the uniform free stream (setParams must have been called). */
  reset(): void {
    if (!this.lastParams) throw new Error('setParams before reset');
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.initPipe);
    pass.setBindGroup(0, this.bgInit);
    pass.dispatchWorkgroups(this.wgx, this.wgy);
    pass.end();
    enc.clearBuffer(this.counter);
    enc.clearBuffer(this.accum);
    this.device.queue.submit([enc.finish()]);
    this.parity = 0;
    this.steps = 0;
  }

  step(enc: GPUCommandEncoder, count: number): void {
    if (count <= 0) return;
    const pass = enc.beginComputePass();
    for (let k = 0; k < count; k++) {
      pass.setPipeline(this.stepPipe);
      pass.setBindGroup(0, this.parity ? this.bgBA : this.bgAB);
      pass.dispatchWorkgroups(this.wgx, this.wgy);
      pass.setPipeline(this.reducePipe);
      pass.setBindGroup(0, this.bgReduce);
      pass.dispatchWorkgroups(1);
      this.parity ^= 1;
    }
    pass.end();
    this.steps += count;
  }

  /** Derived scalar fields for visualization (slice field and volume field). */
  computeFields(enc: GPUCommandEncoder, sliceField: number, volField: number, U: number, lref: number, wall: V3): void {
    const buf = new ArrayBuffer(64);
    new Uint32Array(buf).set([this.nx, this.ny, this.nz, this.n, this.wgx, sliceField, volField, 0]);
    new Float32Array(buf).set([U, lref, 0, 0, ...wall, 0], 8);
    this.device.queue.writeBuffer(this.fieldParams, 0, buf);
    enc.clearBuffer(this.bricks);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.fillPipe);
    pass.setBindGroup(0, this.bgFill);
    pass.dispatchWorkgroups(this.wgx, this.wgy);
    pass.setPipeline(this.fieldsPipe);
    pass.setBindGroup(0, this.bgFields);
    pass.dispatchWorkgroups(this.wgx, this.wgy);
    pass.end();
  }

  /** Copies the accumulated force to the staging buffer (if free) and restarts accumulation. */
  requestForces(enc: GPUCommandEncoder): void {
    if (this.stagingState !== 'idle') return;
    enc.copyBufferToBuffer(this.accum, 0, this.staging, 0, 32);
    enc.clearBuffer(this.accum);
    this.stagingState = 'copied';
  }

  /** Call after submit: resolves with the force sample if one was copied this frame. */
  async readForces(): Promise<ForceSample | null> {
    if (this.stagingState !== 'copied') return null;
    this.stagingState = 'mapping';
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
    } catch {
      this.stagingState = 'idle';
      return null;
    }
    const d = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    this.stagingState = 'idle';
    const steps = d[4];
    if (steps <= 0) return null;
    return { force: [d[0] / steps, d[1] / steps, d[2] / steps], steps };
  }

  destroy(): void {
    for (const b of [this.fA, this.fB, this.flags, this.macro, this.scal, this.bricks, this.partial, this.accum, this.counter, this.qlinks, this.staging, this.params, this.reduceParams, this.fieldParams]) b.destroy();
  }
}
