/**
 * Headless validation: runs the solver without rendering on reference cases and compares the
 * drag coefficient with experimental values. Usage (browser console, dev server running):
 *   const v = await import('/src/validate.ts'); await v.runValidation()
 */
import { buildDomain } from './geometry/domain';
import { sphere, cylinder } from './geometry/primitives';
import { Solver } from './lbm/solver';

interface Case {
  name: string;
  mesh: Float32Array;
  re: number;
  budget: number;
  /** Reference drag coefficient (experiments). */
  cdRef: number;
  /** Convective times to simulate, and to average over at the end. */
  tStar: number;
  tAvg: number;
}

export const CASES: Case[] = [
  { name: 'Sphère Re 100', mesh: sphere(1, 96), re: 100, budget: 1e6, cdRef: 1.09, tStar: 30, tAvg: 10 },
  { name: 'Sphère Re 1000', mesh: sphere(1, 96), re: 1000, budget: 2e6, cdRef: 0.47, tStar: 30, tAvg: 15 },
  { name: 'Sphère Re 10⁴', mesh: sphere(1, 96), re: 1e4, budget: 4e6, cdRef: 0.40, tStar: 30, tAvg: 15 },
  { name: 'Cylindre fini H/D=4 Re 10⁴', mesh: cylinder(1, 4), re: 1e4, budget: 4e6, cdRef: 0.75, tStar: 40, tAvg: 20 },
];

export async function runCase(device: GPUDevice, c: Case, uLattice = 0.08, cs = 0.14, margins: 'compact' | 'standard' | 'large' | 'huge' = 'standard', model = 0, interpolated = true, log = console.log) {
  const d = buildDomain(c.mesh, { budget: c.budget, margins, ground: 'none', clearance: 0, thicken: false, interpolated });
  const solver = new Solver(device, d.nx, d.ny, d.nz, d.flags, d.links.q);
  const nu = (uLattice * d.lengthCells) / c.re;
  const params = {
    tau: 3 * nu + 0.5,
    smag: cs * cs,
    uin: [uLattice, 0, 0] as [number, number, number],
    turbulence: 0,
    turbK: 0.1,
    wall: [0, 0, 0] as [number, number, number],
    pertPos: [0, 0, 0] as [number, number, number],
    pertRadius: 1,
    pertForce: [0, 0, 0] as [number, number, number],
    pertActive: false,
    model,
  };
  solver.setParams(params);
  solver.reset();
  const total = Math.round((c.tStar * d.lengthCells) / uLattice);
  const avgFrom = total - Math.round((c.tAvg * d.lengthCells) / uLattice);
  const q = 0.5 * uLattice * uLattice * d.frontalArea;
  const t0 = performance.now();
  let sum = 0, sum2 = 0, n = 0;
  const batch = 100;
  for (let s = 0; s < total; s += batch) {
    const enc = device.createCommandEncoder();
    solver.step(enc, batch);
    solver.requestForces(enc);
    device.queue.submit([enc.finish()]);
    const f = await solver.readForces();
    if (f && s >= avgFrom) {
      sum += f.force[0] / q;
      sum2 += (f.force[0] / q) ** 2;
      n++;
    }
    if (f && !Number.isFinite(f.force[0])) break;
    (globalThis as { __progress?: unknown }).__progress = { case: c.name, step: s, total, cd: f ? f.force[0] / q : null, mlups: Math.round((s * solver.n) / (performance.now() - t0) / 1e3) };
  }
  await device.queue.onSubmittedWorkDone();
  const sec = (performance.now() - t0) / 1000;
  const cd = sum / n;
  const res = {
    name: c.name,
    cs,
    model,
    interpolated,
    links: `${d.links.links} (${d.links.misses} sans impact)`,
    margins,
    blockage: +d.blockage.toFixed(3),
    grid: `${d.nx}x${d.ny}x${d.nz}`,
    D: +d.lengthCells.toFixed(1),
    tau: +params.tau.toFixed(5),
    cd: +cd.toFixed(3),
    std: +Math.sqrt(Math.max(sum2 / n - cd * cd, 0)).toFixed(3),
    cdRef: c.cdRef,
    err: `${(((cd - c.cdRef) / c.cdRef) * 100).toFixed(1)} %`,
    mlups: Math.round((total * solver.n) / sec / 1e6),
    seconds: +sec.toFixed(1),
  };
  log(res);
  solver.destroy();
  return res;
}

export async function device() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const lim = adapter!.limits;
  return adapter!.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
  });
}

export async function runValidation(which?: number[]) {
  const dev = await device();
  const out = [];
  for (const [i, c] of CASES.entries()) if (!which || which.includes(i)) out.push(await runCase(dev, c));
  dev.destroy();
  return out;
}

/** Runs a case and returns velocity / density profiles (diagnostics). */
export async function probeCase(device: GPUDevice, c: Case, steps: number, uLattice = 0.08, cs = 0.14) {
  const d = buildDomain(c.mesh, { budget: c.budget, margins: 'standard', ground: 'none', clearance: 0, thicken: false, interpolated: true });
  const solver = new Solver(device, d.nx, d.ny, d.nz, d.flags, d.links.q);
  const nu = (uLattice * d.lengthCells) / c.re;
  solver.setParams({
    tau: 3 * nu + 0.5, smag: cs * cs, uin: [uLattice, 0, 0], turbulence: 0, turbK: 0.1, wall: [0, 0, 0],
    pertPos: [0, 0, 0], pertRadius: 1, pertForce: [0, 0, 0], pertActive: false,
  });
  solver.reset();
  for (let s = 0; s < steps; s += 100) {
    const enc = device.createCommandEncoder();
    solver.step(enc, 100);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const read = device.createBuffer({ size: 16 * solver.n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(solver.macro, 0, read, 0, 16 * solver.n);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const m = new Float32Array(read.getMappedRange().slice(0));
  read.destroy();
  const at = (x: number, y: number, z: number) => {
    const i = 4 * (x + d.nx * (y + d.ny * z));
    return [m[i], m[i + 1] / uLattice, m[i + 2] / uLattice, m[i + 3] / uLattice].map((v) => +v.toFixed(3));
  };
  const cy = Math.round(d.ny / 2), cz = Math.round(d.nz / 2);
  const axis: Record<number, number[]> = {};
  for (const x of [0, 1, 2, 5, 10, Math.floor(d.bodyMin[0]) - 3, Math.floor(d.bodyMin[0]) - 1, d.nx - 10, d.nx - 2, d.nx - 1]) axis[x] = at(x, cy, cz);
  const side: Record<number, number[]> = {};
  const xm = Math.round((d.bodyMin[0] + d.bodyMax[0]) / 2);
  for (const y of [0, 1, 2, 5, Math.floor(d.bodyMin[1]) - 2]) side[y] = at(xm, y, cz);
  // mass flux through planes
  const flux = (x: number) => {
    let s = 0;
    for (let z = 1; z < d.nz - 1; z++) for (let y = 1; y < d.ny - 1; y++) { const i = 4 * (x + d.nx * (y + d.ny * z)); s += m[i] * m[i + 1]; }
    return +(s / ((d.ny - 2) * (d.nz - 2) * uLattice)).toFixed(4);
  };
  solver.destroy();
  return { grid: [d.nx, d.ny, d.nz], body: [d.bodyMin, d.bodyMax].map((v) => v.map((a) => +a.toFixed(1))), axis, side, flux: { inlet: flux(1), body: flux(xm), outlet: flux(d.nx - 2) } };
}
