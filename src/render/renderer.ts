import type { CameraMatrices } from './camera';
import { OVERLAY_WGSL, SCENE_WGSL } from './shaders';
import type { V3 } from './math';

export interface ViewSettings {
  dims: V3;
  U: number;
  slice: { show: boolean; axis: number; pos: number; min: number; max: number; diverging: boolean };
  particles: { show: boolean; count: number; size: number; speedMax: number };
  volume: { mode: 0 | 1 | 2; threshold: number; max: number; density: number };
  surfaceCp: boolean;
  cpRange: [number, number];
  ground: boolean;
  pert: { pos: V3; radius: number } | null;
}

export interface SceneBuffers {
  macro: GPUBuffer;
  flags: GPUBuffer;
  scal: GPUBuffer;
  particles: GPUBuffer;
  bricks: GPUBuffer;
}

const DEPTH = 'depth32float';

export class Renderer {
  private ctx: GPUCanvasContext;
  private ubo: GPUBuffer;
  private bgl: GPUBindGroupLayout;
  private bglDepth: GPUBindGroupLayout;
  private bg: GPUBindGroup | null = null;
  private bgDepth: GPUBindGroup | null = null;
  private depth: GPUTexture | null = null;
  private pipes: Record<string, GPURenderPipeline> = {};
  private mesh: GPUBuffer | null = null;
  private meshVerts = 0;
  private uboData = new Float32Array(72);

  constructor(private device: GPUDevice, private canvas: HTMLCanvasElement, format: GPUTextureFormat) {
    this.ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    this.ctx.configure({ device, format, alphaMode: 'opaque' });
    this.ubo = device.createBuffer({ size: 288, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const ro = { type: 'read-only-storage' } as const;
    this.bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 5].map((binding) => ({ binding, visibility: VF, buffer: ro })),
      ],
    });
    this.bglDepth = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } }],
    });
    const scene = device.createShaderModule({ code: SCENE_WGSL, label: 'scene' });
    const overlay = device.createShaderModule({ code: OVERLAY_WGSL, label: 'overlay' });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    const layout2 = device.createPipelineLayout({ bindGroupLayouts: [this.bgl, this.bglDepth] });
    const depthState: GPUDepthStencilState = { format: DEPTH, depthWriteEnabled: true, depthCompare: 'less' };
    const make = (
      name: string,
      vs: string,
      fs: string,
      topology: GPUPrimitiveTopology,
      opts: { buffers?: GPUVertexBufferLayout[]; overlay?: boolean; blend?: GPUBlendState } = {},
    ) => {
      this.pipes[name] = device.createRenderPipeline({
        label: name,
        layout: opts.overlay ? layout2 : layout,
        vertex: { module: opts.overlay ? overlay : scene, entryPoint: vs, buffers: opts.buffers ?? [] },
        fragment: { module: opts.overlay ? overlay : scene, entryPoint: fs, targets: [{ format, blend: opts.blend }] },
        primitive: { topology },
        depthStencil: opts.overlay ? undefined : depthState,
      });
    };
    const premul: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    };
    make('box', 'vsBox', 'fsLine', 'line-list');
    make('ground', 'vsGround', 'fsGround', 'triangle-list');
    make('slice', 'vsSlice', 'fsSlice', 'triangle-list');
    make('object', 'vsObject', 'fsObject', 'triangle-list', {
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }],
    });
    make('particles', 'vsParticle', 'fsParticle', 'triangle-list');
    make('iso', 'vsFull', 'fsIso', 'triangle-list');
    make('fog', 'vsFull', 'fsFog', 'triangle-list', { overlay: true, blend: premul });
    make('pert', 'vsPert', 'fsPert', 'triangle-list', { overlay: true, blend: premul });
  }

  setScene(buffers: SceneBuffers, meshVertices: Float32Array<ArrayBuffer>): void {
    this.bg = this.device.createBindGroup({
      layout: this.bgl,
      entries: [this.ubo, buffers.macro, buffers.flags, buffers.scal, buffers.particles, buffers.bricks].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    this.mesh?.destroy();
    this.mesh = this.device.createBuffer({ size: Math.max(24, meshVertices.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.mesh, 0, meshVertices);
    this.meshVerts = meshVertices.length / 6;
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.depth) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.depth?.destroy();
      this.depth = this.device.createTexture({
        size: [w, h],
        format: DEPTH,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.bgDepth = this.device.createBindGroup({ layout: this.bglDepth, entries: [{ binding: 0, resource: this.depth.createView() }] });
    }
  }

  render(enc: GPUCommandEncoder, cam: CameraMatrices, s: ViewSettings): void {
    if (!this.bg) return;
    this.resize();
    const u = this.uboData;
    u.set(cam.viewProj, 0);
    u.set(cam.invViewProj, 16);
    u.set([...cam.eye, s.particles.size], 32);
    u.set([...s.dims, s.U], 36);
    u.set([s.slice.axis, s.slice.pos, s.slice.diverging ? 1 : 0, 0], 40);
    u.set([s.slice.min, s.slice.max, s.volume.threshold, s.volume.max], 44);
    u.set([this.canvas.width, this.canvas.height, s.ground ? 1 : 0, s.particles.speedMax], 48);
    u.set(s.pert ? [...s.pert.pos, s.pert.radius] : [0, 0, 0, -1], 52);
    u.set([s.surfaceCp ? 1 : 0, s.cpRange[0], s.cpRange[1], s.volume.density], 56);
    u.set([0.2, 1.4, s.volume.mode, 0], 60);
    u.set([...cam.right, 0], 64);
    u.set([...cam.up, 0], 68);
    this.device.queue.writeBuffer(this.ubo, 0, u);

    const view = this.ctx.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view, clearValue: { r: 0.075, g: 0.082, b: 0.098, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setBindGroup(0, this.bg);
    pass.setPipeline(this.pipes.box);
    pass.draw(24);
    if (s.ground) {
      pass.setPipeline(this.pipes.ground);
      pass.draw(6);
    }
    if (this.mesh && this.meshVerts) {
      pass.setPipeline(this.pipes.object);
      pass.setVertexBuffer(0, this.mesh);
      pass.draw(this.meshVerts);
    }
    if (s.slice.show) {
      pass.setPipeline(this.pipes.slice);
      pass.draw(6);
    }
    if (s.particles.show && s.particles.count > 0) {
      pass.setPipeline(this.pipes.particles);
      pass.draw(6, s.particles.count);
    }
    if (s.volume.mode === 1) {
      pass.setPipeline(this.pipes.iso);
      pass.draw(3);
    }
    pass.end();

    if (s.volume.mode === 2 || s.pert) {
      const p2 = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: 'load', storeOp: 'store' }] });
      p2.setBindGroup(0, this.bg);
      p2.setBindGroup(1, this.bgDepth!);
      if (s.volume.mode === 2) {
        p2.setPipeline(this.pipes.fog);
        p2.draw(3);
      }
      if (s.pert) {
        p2.setPipeline(this.pipes.pert);
        p2.draw(6);
      }
      p2.end();
    }
  }
}
