import { initGpu, type Gpu } from './gpu/device';
import { buildDomain, type Domain, type GroundMode, type MarginPreset } from './geometry/domain';
import { loadModelFile, type UpAxis } from './geometry/loaders';
import { applyMatrix3, rotationMatrix, withFlatNormals, yUpToZUp } from './geometry/mesh';
import { ahmedBody, box, cylinder, nacaWing, simpleCar, sphere } from './geometry/primitives';
import { Solver } from './lbm/solver';
import { SLICE_FIELDS } from './lbm/shaders';
import { FLUIDS, flowScales, formatReynolds, formatSI, type FluidId } from './lbm/units';
import { OrbitCamera } from './render/camera';
import { add, dot, length, scale, sub, type V3 } from './render/math';
import { Particles, type RakeMode } from './render/particles';
import { Renderer } from './render/renderer';
import { drawChart } from './ui/chart';
import { drawColorbar } from './ui/colormap';
import { Measurements } from './ui/measurements';

// ------------------------------------------------------------------ presets

interface Preset {
  id: string;
  label: string;
  make: () => Float32Array;
  length: number;
  speed: number;
  ground: GroundMode;
  clearance?: number;
  pitch?: number;
  refArea?: 'frontal' | 'planform';
}

const PRESETS: Preset[] = [
  { id: 'sphere', label: 'Sphère', make: () => sphere(1), length: 0.2, speed: 20, ground: 'none' },
  { id: 'cylinder', label: 'Cylindre (axe vertical, H = 4 D)', make: () => cylinder(1, 4), length: 0.05, speed: 3, ground: 'none' },
  { id: 'naca2412', label: 'Aile NACA 2412 (allongement 4)', make: () => nacaWing('2412', 1, 4), length: 0.3, speed: 30, ground: 'none', pitch: 6, refArea: 'planform' },
  { id: 'naca0012', label: 'Aile NACA 0012 (allongement 4)', make: () => nacaWing('0012', 1, 4), length: 0.3, speed: 30, ground: 'none', pitch: 8, refArea: 'planform' },
  { id: 'ahmed25', label: "Corps d'Ahmed (lunette 25°)", make: () => ahmedBody(25), length: 1.044, speed: 40, ground: 'fixed', clearance: 0 },
  { id: 'ahmed35', label: "Corps d'Ahmed (lunette 35°)", make: () => ahmedBody(35), length: 1.044, speed: 40, ground: 'fixed', clearance: 0 },
  { id: 'car', label: 'Voiture simplifiée', make: simpleCar, length: 4.2, speed: 30, ground: 'moving', clearance: 0 },
  { id: 'cube', label: 'Cube', make: () => box(1, 1, 1), length: 0.1, speed: 10, ground: 'none' },
  { id: 'plate', label: "Plaque face au vent", make: () => box(0.04, 1, 1), length: 0.02, speed: 10, ground: 'none' },
];

const BUDGETS = [0.25e6, 0.5e6, 1e6, 2e6, 4e6, 8e6, 16e6, 32e6];

// ------------------------------------------------------------------ DOM helpers

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');

function bindSlider(id: string, show: (v: number) => string, onInput: (v: number) => void, onChange?: (v: number) => void): HTMLInputElement {
  const el = $<HTMLInputElement>(id);
  const out = document.getElementById('o-' + id);
  const refresh = () => out && (out.textContent = show(+el.value));
  el.addEventListener('input', () => {
    refresh();
    onInput(+el.value);
  });
  if (onChange) el.addEventListener('change', () => onChange(+el.value));
  refresh();
  (el as HTMLInputElement & { refresh?: () => void }).refresh = refresh;
  return el;
}

function setSlider(el: HTMLInputElement, v: number): void {
  el.value = String(v);
  (el as HTMLInputElement & { refresh?: () => void }).refresh?.();
}

function kv(el: HTMLElement, rows: [string, string, boolean?][]): void {
  el.innerHTML = rows.map(([k, v, warn]) => `<dt>${k}</dt><dd${warn ? ' class="warn"' : ''}>${v}</dd>`).join('');
}

// ------------------------------------------------------------------ app

class App {
  private camera = new OrbitCamera();
  private renderer: Renderer;
  private solver: Solver | null = null;
  private particles: Particles | null = null;
  private domain: Domain | null = null;
  private stats = new Measurements();

  // model
  private source: Float32Array = new Float32Array();
  private sourceName = '';
  private sourceUp: UpAxis = 'z';
  private yaw = 0;
  private pitch = 0;
  private roll = 0;
  private ground: GroundMode = 'none';
  private clearance = 0;
  private thicken = false;
  private interpolated = true;
  private budget = 1e6;
  private margins: MarginPreset = 'standard';

  // flow
  private fluid: FluidId = 'air';
  private speed = 30;
  private length = 1;
  private uLattice = 0.08;
  private smagorinsky = 0.14;
  private turbulence = 0;
  private gustAmplitude = 0.35;
  private gustStart = -1;
  private running = true;
  private stepsPerFrame = 0; // 0 = auto
  private autoSteps = 4;

  // forcing
  private tool: 'camera' | 'blow' = 'camera';
  private pertRadius = 6;
  private pertStrength = 1;
  private drag: { mode: 'orbit' | 'pan' | 'blow'; x: number; y: number } | null = null;
  private pert = { active: false, sx: 0, sy: 0, prev: null as V3 | null, vel: [0, 0, 0] as V3, pos: [0, 0, 0] as V3, force: [0, 0, 0] as V3 };

  // view
  private slice = { show: true, field: 0, axis: 1, pos: 0.5, min: 0, max: 1.5 };
  private smoke = { show: true, mode: 'horizontal' as RakeMode, y: 0.5, z: 0.5, w: 0.5, nozzles: 24, count: 131072, size: 2 };
  private volMode: 0 | 1 | 2 = 0;
  private volThr = 0.35;
  private surfaceCp = true;
  private refArea: 'frontal' | 'planform' = 'frontal';
  private avgWindow = 10;

  // perf
  private lastFrame = performance.now();
  private frameTimes: number[] = [];
  private perfWindow = { t: performance.now(), steps: 0, frames: 0 };
  private building = false;
  private rebuildTimer = 0;
  private lastUiUpdate = 0;

  constructor(private gpu: Gpu) {
    this.renderer = new Renderer(gpu.device, $<HTMLCanvasElement>('view'), gpu.format);
    this.setupUi();
    this.setupInput();
    this.loadPreset(PRESETS[0]);
    requestAnimationFrame(this.frame);
  }

  // ---------------------------------------------------------------- model and domain

  private loadPreset(p: Preset): void {
    this.source = p.make();
    this.sourceName = p.label;
    this.sourceUp = 'z';
    $<HTMLSelectElement>('up-axis').value = 'z';
    this.yaw = 0;
    this.pitch = p.pitch ?? 0;
    this.roll = 0;
    this.ground = p.ground;
    this.clearance = p.clearance ?? 0.15;
    this.length = p.length;
    this.speed = p.speed;
    this.refArea = p.refArea ?? 'frontal';
    this.syncModelControls();
    this.rebuild(true);
  }

  private async importFile(file: File): Promise<void> {
    this.setBusy(`Lecture de ${file.name}…`);
    try {
      const m = await loadModelFile(file);
      this.source = m.mesh.positions;
      this.sourceName = m.mesh.name;
      this.sourceUp = m.upAxis;
      $<HTMLSelectElement>('up-axis').value = m.upAxis;
      $<HTMLSelectElement>('preset').value = '';
      this.yaw = 0;
      this.pitch = 0;
      this.roll = 0;
      this.syncModelControls();
      this.toast(`« ${file.name} » importé (${(this.source.length / 9).toLocaleString('fr-FR')} triangles). Orientez-le pour que l'avant regarde vers l'entrée (−X) et indiquez sa longueur réelle.`, 'info');
      await this.rebuild(true);
    } catch (e) {
      this.setBusy(null);
      this.toast((e as Error).message);
    }
  }

  private modelPositions(): Float32Array {
    let p = this.sourceUp === 'y' ? yUpToZUp(this.source) : this.source;
    // centre on the bounding box before rotating
    let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], p[i + a]);
      max[a] = Math.max(max[a], p[i + a]);
    }
    const c = [0, 1, 2].map((a) => (min[a] + max[a]) / 2);
    const centred = new Float32Array(p.length);
    for (let i = 0; i < p.length; i += 3) for (let a = 0; a < 3; a++) centred[i + a] = p[i + a] - c[a];
    return applyMatrix3(centred, rotationMatrix(this.yaw, this.pitch, this.roll));
  }

  private scheduleRebuild(): void {
    clearTimeout(this.rebuildTimer);
    this.rebuildTimer = window.setTimeout(() => this.rebuild(false), 150);
  }

  private async rebuild(fitCamera: boolean): Promise<void> {
    if (this.building) {
      this.scheduleRebuild();
      return;
    }
    this.building = true;
    this.setBusy('Voxelisation de la maquette…');
    await new Promise((r) => setTimeout(r, 30));
    try {
      const budget = Math.min(this.budget, this.gpu.maxCells);
      const domain = buildDomain(this.modelPositions(), {
        budget,
        margins: this.margins,
        ground: this.ground,
        clearance: this.clearance,
        thicken: this.thicken,
        interpolated: this.interpolated,
      });
      if (domain.solidCells === 0) throw new Error("La maquette est trop petite ou trop fine pour cette résolution : augmentez la résolution ou cochez « Épaissir les surfaces ».");
      this.setBusy('Allocation sur le GPU…');
      await new Promise((r) => setTimeout(r, 10));
      this.solver?.destroy();
      this.particles?.destroy();
      this.solver = null;
      this.particles = null;
      this.gpu.device.pushErrorScope('out-of-memory');
      const solver = new Solver(this.gpu.device, domain.nx, domain.ny, domain.nz, domain.flags, domain.links.q);
      const particles = new Particles(this.gpu.device, 524288, solver.macro, solver.flags);
      const err = await this.gpu.device.popErrorScope();
      if (err) {
        solver.destroy();
        particles.destroy();
        throw new Error(`Mémoire GPU insuffisante pour ${(domain.nx * domain.ny * domain.nz / 1e6).toFixed(1)} M cellules : choisissez une résolution plus faible.`);
      }
      this.domain = domain;
      this.solver = solver;
      this.particles = particles;
      this.renderer.setScene(
        { macro: solver.macro, flags: solver.flags, scal: solver.scal, particles: particles.buffer, bricks: solver.bricks },
        withFlatNormals(domain.tris),
      );
      this.solver.setParams(this.stepParams());
      this.solver.reset();
      this.stats.clear();
      this.gustStart = -1;
      this.placeRakeAndSlice();
      if (fitCamera) this.fitCamera();
      this.updateInfo();
      if (domain.openFraction > 0.01 && !this.thicken) {
        this.toast(`Le maillage semble ouvert (${(domain.openFraction * 100).toFixed(1)} % de rayons ambigus). Si la maquette paraît trouée, cochez « Épaissir les surfaces ».`);
      }
    } catch (e) {
      this.toast((e as Error).message);
    } finally {
      this.building = false;
      this.setBusy(null);
    }
  }

  private placeRakeAndSlice(): void {
    const d = this.domain!;
    const c = scale(add(d.bodyMin, d.bodyMax), 0.5);
    this.smoke.y = c[1] / d.ny;
    this.smoke.z = c[2] / d.nz;
    setSlider($('smoke-y'), this.smoke.y);
    setSlider($('smoke-z'), this.smoke.z);
    this.setSliceAxis(this.slice.axis);
  }

  private setSliceAxis(axis: number): void {
    this.slice.axis = axis;
    document.querySelectorAll<HTMLButtonElement>('.segmented button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.axis! === axis)));
    const d = this.domain;
    if (d) {
      const n = [d.nx, d.ny, d.nz][axis];
      const c = scale(add(d.bodyMin, d.bodyMax), 0.5);
      // transverse slice: just behind the body, in the near wake
      const p = axis === 0 ? d.bodyMax[0] + 0.3 * (d.bodyMax[0] - d.bodyMin[0]) : c[axis];
      this.slice.pos = p / n;
      setSlider($('slice-pos'), this.slice.pos);
    }
  }

  private fitCamera(): void {
    const d = this.domain!;
    const c = scale(add(d.bodyMin, d.bodyMax), 0.5);
    const span = Math.max(d.bodyMax[0] - d.bodyMin[0], d.bodyMax[1] - d.bodyMin[1], d.bodyMax[2] - d.bodyMin[2]);
    this.camera.fit([c[0] + 0.6 * span, c[1], c[2]], Math.max(span * 1.6, Math.min(d.nx, d.ny + d.nz) * 0.45));
    this.camera.yaw = -2.35;
    this.camera.pitch = 0.38;
  }

  // ---------------------------------------------------------------- physics parameters

  private scales() {
    return flowScales({
      fluid: this.fluid,
      speed: this.speed,
      length: this.length,
      uLattice: this.uLattice,
      lengthCells: this.domain?.lengthCells ?? 1,
    });
  }

  private gustFactor(): number {
    if (this.gustStart < 0 || !this.solver || !this.domain) return 1;
    const duration = (2 * this.domain.lengthCells) / this.uLattice;
    const s = (this.solver.steps - this.gustStart) / duration;
    if (s >= 1) {
      this.gustStart = -1;
      return 1;
    }
    return 1 + this.gustAmplitude * Math.sin(Math.PI * s) ** 2;
  }

  private stepParams() {
    const sc = this.scales();
    const U = this.uLattice * this.gustFactor();
    const d = this.domain;
    const F = (0.6 * this.pertStrength * this.uLattice * this.uLattice) / this.pertRadius;
    return {
      tau: Math.max(sc.tau, 0.5 + 1e-6),
      smag: this.smagorinsky * this.smagorinsky,
      uin: [U, 0, 0] as V3,
      turbulence: this.turbulence / 100,
      turbK: (2 * Math.PI) / Math.max(4, (d?.lengthCells ?? 32) * 0.5),
      wall: [this.ground === 'moving' ? this.uLattice : 0, 0, 0] as V3,
      pertPos: this.pert.pos,
      pertRadius: this.pertRadius,
      pertForce: scale(this.pert.force, F),
      pertActive: this.pert.active,
    };
  }

  // ---------------------------------------------------------------- frame loop

  private frame = (now: number) => {
    requestAnimationFrame(this.frame);
    const dtFrame = now - this.lastFrame;
    this.lastFrame = now;
    if (!this.solver || !this.domain || !this.particles || this.building) return;
    const solver = this.solver;
    const d = this.domain;

    this.frameTimes.push(dtFrame);
    if (this.frameTimes.length > 12) this.frameTimes.shift();
    if (this.stepsPerFrame === 0 && this.frameTimes.length >= 12 && this.running) {
      const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
      if (avg < 19) this.autoSteps = Math.min(80, Math.ceil(this.autoSteps * 1.12));
      else if (avg > 26) this.autoSteps = Math.max(1, Math.floor(this.autoSteps * 0.85));
      this.frameTimes.length = 0;
    }
    const spf = this.running ? this.stepsPerFrame || this.autoSteps : 0;

    const aspect = this.renderer ? $('view').clientWidth / Math.max(1, $('view').clientHeight) : 1;
    const cam = this.camera.matrices(aspect);
    this.updatePerturbation(cam);

    const enc = this.gpu.device.createCommandEncoder();
    solver.setParams(this.stepParams());
    solver.step(enc, spf);
    const sliceField = SLICE_FIELDS[this.slice.field];
    solver.computeFields(enc, sliceField.id, this.volMode, this.uLattice, d.lengthCells, [this.ground === 'moving' ? this.uLattice : 0, 0, 0]);
    const rakeHalf = this.smoke.w * 0.5;
    this.particles.update(
      enc,
      [d.nx, d.ny, d.nz],
      spf,
      {
        mode: this.smoke.mode,
        y: this.smoke.y * d.ny,
        z: this.smoke.z * d.nz,
        halfY: rakeHalf * (d.ny - 2),
        halfZ: rakeHalf * (d.nz - 2),
        nozzles: this.smoke.nozzles,
      },
      this.smoke.count,
      (2.5 * d.nx) / this.uLattice,
    );
    const vt = this.volumeThreshold();
    this.renderer.render(enc, cam, {
      dims: [d.nx, d.ny, d.nz],
      U: this.uLattice,
      slice: {
        show: this.slice.show,
        axis: this.slice.axis,
        pos: this.slice.pos * [d.nx, d.ny, d.nz][this.slice.axis],
        min: this.slice.min,
        max: this.slice.max,
        diverging: sliceField.diverging,
      },
      particles: { show: this.smoke.show, count: this.smoke.count, size: this.smoke.size * Math.min(window.devicePixelRatio || 1, 2) * 0.5, speedMax: 1.5 },
      volume: { mode: this.volMode, threshold: vt, max: vt * 6, density: 0.35 },
      surfaceCp: this.surfaceCp,
      cpRange: [-1.5, 1],
      ground: this.ground !== 'none',
      pert: this.pert.active ? { pos: this.pert.pos, radius: this.pertRadius } : null,
    });
    if (spf > 0) solver.requestForces(enc);
    this.gpu.device.queue.submit([enc.finish()]);
    solver.readForces().then((s) => s && solver === this.solver && this.onForces(s.force));

    this.perfWindow.steps += spf;
    this.perfWindow.frames++;
    if (now - this.perfWindow.t > 1000) {
      const sec = (now - this.perfWindow.t) / 1000;
      $('r-mlups').textContent = ((this.perfWindow.steps * solver.n) / sec / 1e6).toFixed(0);
      $('r-fps').textContent = (this.perfWindow.frames / sec).toFixed(0);
      $('r-spf').textContent = String(spf);
      this.perfWindow = { t: now, steps: 0, frames: 0 };
    }
    if (now - this.lastUiUpdate > 150) {
      this.lastUiUpdate = now;
      this.updateReadouts();
    }
  };

  private volumeThreshold(): number {
    return this.volMode === 2 ? 10 ** (0.3 + 2.2 * this.volThr) : 10 ** (-0.5 + 4 * this.volThr);
  }

  private onForces(f: V3): void {
    const d = this.domain!;
    if (!f.every(Number.isFinite) || Math.abs(f[0]) > 1e6) {
      this.running = false;
      this.updatePauseButton();
      this.toast("Instabilité numérique détectée : l'écoulement a divergé. Réduisez la vitesse réseau ou le nombre de Reynolds, augmentez la résolution ou Cs, puis cliquez sur « Réinitialiser ».");
      return;
    }
    const A = this.refArea === 'frontal' ? d.frontalArea : d.planformArea;
    const q = 0.5 * this.uLattice * this.uLattice * Math.max(A, 1);
    this.stats.add({ step: this.solver!.steps, cd: f[0] / q, cl: f[2] / q, cs: f[1] / q });
  }

  private updateReadouts(): void {
    const d = this.domain;
    const solver = this.solver;
    if (!d || !solver) return;
    const sc = this.scales();
    const tstarPerStep = this.uLattice / d.lengthCells;
    $('r-steps').textContent = solver.steps.toLocaleString('fr-FR');
    $('r-tstar').textContent = (solver.steps * tstarPerStep).toFixed(2);
    $('r-tphys').textContent = formatSI(solver.steps * sc.dt, 's');

    const last = this.stats.samples[this.stats.samples.length - 1];
    const w = this.stats.window(this.avgWindow / tstarPerStep, d.lengthCells, this.uLattice);
    $('h-cd').textContent = fmt(w.cd);
    $('h-cl').textContent = fmt(w.cl);
    $('h-cs').textContent = fmt(w.cs);
    $('h-sub').innerHTML = `moyenne sur ${this.avgWindow} t* · Re ${formatReynolds(sc.reynolds)}`;
    $('m-cd').textContent = fmt(last?.cd);
    $('m-cl').textContent = fmt(last?.cl);
    $('m-cs').textContent = fmt(last?.cs);
    $('m-cd-avg').textContent = fmt(w.cd);
    $('m-cl-avg').textContent = fmt(w.cl);
    $('m-cs-avg').textContent = fmt(w.cs);
    $('m-cd-std').textContent = fmt(w.cdStd);
    $('m-cl-std').textContent = fmt(w.clStd);
    const A = (this.refArea === 'frontal' ? d.frontalArea : d.planformArea) * sc.dx * sc.dx;
    const qd = 0.5 * FLUIDS[this.fluid].rho * this.speed * this.speed;
    $('m-drag').textContent = formatSI(w.cd * qd * A, 'N');
    $('m-lift').textContent = formatSI(w.cl * qd * A, 'N');
    $('m-ld').textContent = Math.abs(w.cd) > 1e-6 ? fmt(w.cl / w.cd, 2) : '—';
    $('m-st').textContent = fmt(w.strouhal, 3);
    $('m-scx').textContent = `${formatSI(w.cd * A, 'm²')}`;
    $('m-area').textContent = `${formatSI(A, 'm²')} (${this.refArea === 'frontal' ? 'frontale' : 'alaire'})`;
    drawChart($<HTMLCanvasElement>('chart'), this.stats.samples, tstarPerStep, Math.max(this.avgWindow * 2, 10));
  }

  private updateInfo(): void {
    const d = this.domain;
    if (!d) return;
    const cells = d.nx * d.ny * d.nz;
    kv($('model-info'), [
      ['Maquette', this.sourceName],
      ['Triangles', (this.source.length / 9).toLocaleString('fr-FR')],
      ['Longueur', `${d.lengthCells.toFixed(1)} cellules`],
      ['Cellules solides', d.solidCells.toLocaleString('fr-FR')],
      ...(d.openFraction > 0.001 ? [['Maillage ouvert', `${(d.openFraction * 100).toFixed(1)} %`, true] as [string, string, boolean]] : []),
    ]);
    kv($('domain-info'), [
      ['Grille', `${d.nx} × ${d.ny} × ${d.nz}`],
      ['Cellules', `${(cells / 1e6).toFixed(2)} M`],
      ['Mémoire GPU', `${(Solver.bytesFor(cells) / 2 ** 20).toFixed(0)} Mo`],
      ['Blocage', `${(d.blockage * 100).toFixed(1)} %`, d.blockage > 0.1],
    ]);
    this.updateFlowInfo();
  }

  private updateFlowInfo(): void {
    if (!this.domain) return;
    const sc = this.scales();
    const lesNote = sc.cellReynolds > 2 ? 'LES (couche limite non résolue)' : sc.cellReynolds > 0.5 ? 'LES / DNS' : 'DNS (résolu)';
    kv($('flow-info'), [
      ['Reynolds', formatReynolds(sc.reynolds)],
      ['Régime numérique', lesNote, sc.cellReynolds > 50],
      ['Δx', formatSI(sc.dx, 'm')],
      ['Δt', formatSI(sc.dt, 's')],
      ['τ (relaxation)', sc.tau.toFixed(6), sc.tau < 0.5005],
      ['Mach réseau', sc.mach.toFixed(3), sc.mach > 0.2],
    ]);
    $('speed-note').textContent = `${(this.speed * 3.6).toFixed(1)} km/h · ${FLUIDS[this.fluid].label}`;
    this.updateLegend();
  }

  private updateLegend(): void {
    const f = SLICE_FIELDS[this.slice.field];
    $('legend-title').textContent = f.label;
    $('legend-min').textContent = String(+this.slice.min.toPrecision(3));
    $('legend-max').textContent = String(+this.slice.max.toPrecision(3));
    drawColorbar($<HTMLCanvasElement>('legend-bar'), f.diverging);
  }

  // ---------------------------------------------------------------- mouse forcing

  private updatePerturbation(cam: ReturnType<OrbitCamera['matrices']>): void {
    if (!this.drag || this.drag.mode !== 'blow' || !this.domain) {
      this.pert.active = false;
      this.pert.prev = null;
      return;
    }
    const canvas = $('view');
    const r = canvas.getBoundingClientRect();
    const ndcX = ((this.pert.sx - r.left) / r.width) * 2 - 1;
    const ndcY = 1 - ((this.pert.sy - r.top) / r.height) * 2;
    const ray = this.camera.ray(cam, ndcX, ndcY);
    // plane through the orbit target, facing the camera
    const denom = dot(ray.dir, cam.forward);
    if (Math.abs(denom) < 1e-6) return;
    const t = dot(sub(this.camera.target, ray.origin), cam.forward) / denom;
    const hit = add(ray.origin, scale(ray.dir, t));
    const vel: V3 = this.pert.prev ? sub(hit, this.pert.prev) : [0, 0, 0];
    this.pert.prev = hit;
    this.pert.vel = add(scale(this.pert.vel, 0.6), scale(vel, 0.4));
    const sp = length(this.pert.vel);
    this.pert.pos = hit;
    this.pert.active = true;
    if (sp > 0.05) {
      this.pert.force = scale(this.pert.vel, Math.min(sp / 2, 1.5) / sp);
    } else {
      this.pert.force = cam.forward; // held still: blow along the view direction
    }
  }

  // ---------------------------------------------------------------- input

  private setupInput(): void {
    const canvas = $<HTMLCanvasElement>('view');
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      const blow = e.button === 0 && (this.tool === 'blow' || e.ctrlKey);
      this.drag = { mode: blow ? 'blow' : e.button === 0 && !e.shiftKey ? 'orbit' : 'pan', x: e.clientX, y: e.clientY };
      this.pert.sx = e.clientX;
      this.pert.sy = e.clientY;
      this.pert.prev = null;
      this.pert.vel = [0, 0, 0];
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      this.drag.x = e.clientX;
      this.drag.y = e.clientY;
      if (this.drag.mode === 'orbit') this.camera.orbit(dx, dy);
      else if (this.drag.mode === 'pan') {
        const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
        this.camera.pan(dx, dy, canvas.clientHeight, this.camera.matrices(aspect));
      } else {
        this.pert.sx = e.clientX;
        this.pert.sy = e.clientY;
      }
    });
    const end = () => {
      this.drag = null;
      this.pert.active = false;
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.zoom(Math.exp(e.deltaY * 0.0012));
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.code === 'Space') {
        e.preventDefault();
        this.togglePause();
      } else if (e.key === 'r' || e.key === 'R') this.resetFlow();
      else if (e.key === 'g' || e.key === 'G') this.gust();
      else if (e.key === 'c' || e.key === 'C') this.domain && this.fitCamera();
    });

    // drag & drop import
    const drop = $('drop');
    let depth = 0;
    window.addEventListener('dragenter', (e) => {
      if (!e.dataTransfer?.types.includes('Files')) return;
      depth++;
      drop.hidden = false;
    });
    window.addEventListener('dragleave', () => {
      depth = Math.max(0, depth - 1);
      if (!depth) drop.hidden = true;
    });
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => {
      e.preventDefault();
      depth = 0;
      drop.hidden = true;
      const file = e.dataTransfer?.files[0];
      if (file) this.importFile(file);
    });
  }

  private togglePause(): void {
    this.running = !this.running;
    this.updatePauseButton();
  }

  private updatePauseButton(): void {
    $('btn-pause').textContent = this.running ? 'Pause' : 'Reprendre';
  }

  private resetFlow(): void {
    if (!this.solver) return;
    this.solver.setParams(this.stepParams());
    this.solver.reset();
    this.particles?.reset();
    this.stats.clear();
    this.gustStart = -1;
    this.running = true;
    this.updatePauseButton();
  }

  private gust(): void {
    if (this.solver) this.gustStart = this.solver.steps;
  }

  // ---------------------------------------------------------------- panel

  private setupUi(): void {
    const preset = $<HTMLSelectElement>('preset');
    preset.innerHTML = PRESETS.map((p) => `<option value="${p.id}">${p.label}</option>`).join('') + '<option value="" disabled>— fichier importé —</option>';
    preset.addEventListener('change', () => {
      const p = PRESETS.find((q) => q.id === preset.value);
      if (p) this.loadPreset(p);
    });

    const file = $<HTMLInputElement>('file');
    $('btn-import').addEventListener('click', () => file.click());
    file.addEventListener('change', () => {
      if (file.files?.[0]) this.importFile(file.files[0]);
      file.value = '';
    });
    $<HTMLSelectElement>('up-axis').addEventListener('change', (e) => {
      this.sourceUp = (e.target as HTMLSelectElement).value as UpAxis;
      this.rebuild(true);
    });
    const deg = (v: number) => `${v}°`;
    bindSlider('yaw', deg, (v) => (this.yaw = v), () => this.scheduleRebuild());
    bindSlider('pitch', deg, (v) => (this.pitch = v), () => this.scheduleRebuild());
    bindSlider('roll', deg, (v) => (this.roll = v), () => this.scheduleRebuild());
    $<HTMLSelectElement>('ground').addEventListener('change', (e) => {
      this.ground = (e.target as HTMLSelectElement).value as GroundMode;
      this.syncModelControls();
      this.rebuild(true);
    });
    bindSlider('clearance', (v) => `${Math.round(v * 100)} % de la hauteur`, (v) => (this.clearance = v), () => this.scheduleRebuild());
    $<HTMLInputElement>('thicken').addEventListener('change', (e) => {
      this.thicken = (e.target as HTMLInputElement).checked;
      this.rebuild(false);
    });

    const budget = $<HTMLSelectElement>('budget');
    budget.innerHTML = BUDGETS.map((b) => {
      const ok = b <= this.gpu.maxCells;
      const mem = Solver.bytesFor(b) / 2 ** 20;
      const label = `${b >= 1e6 ? b / 1e6 + ' M' : b / 1e3 + ' k'} cellules · ${mem >= 1024 ? (mem / 1024).toFixed(1) + ' Go' : mem.toFixed(0) + ' Mo'}`;
      return `<option value="${b}" ${ok ? '' : 'disabled'}>${label}${ok ? '' : ' (trop pour ce GPU)'}</option>`;
    }).join('');
    budget.value = String(this.budget);
    budget.addEventListener('change', () => {
      this.budget = +budget.value;
      this.rebuild(true);
    });
    $<HTMLSelectElement>('margins').addEventListener('change', (e) => {
      this.margins = (e.target as HTMLSelectElement).value as MarginPreset;
      this.rebuild(true);
    });

    $<HTMLSelectElement>('fluid').addEventListener('change', (e) => {
      this.fluid = (e.target as HTMLSelectElement).value as FluidId;
      this.updateFlowInfo();
    });
    const num = (id: string, set: (v: number) => void) => {
      const el = $<HTMLInputElement>(id);
      el.addEventListener('input', () => {
        const v = parseFloat(el.value);
        if (v > 0 && Number.isFinite(v)) {
          set(v);
          this.updateFlowInfo();
        }
      });
    };
    num('speed', (v) => (this.speed = v));
    num('length', (v) => (this.length = v));
    bindSlider('turb', (v) => `${v} %`, (v) => (this.turbulence = v));
    bindSlider('gust', (v) => `+${Math.round(v * 100)} %`, (v) => (this.gustAmplitude = v));
    bindSlider('ulb', (v) => `${v.toFixed(3)} (Ma ${(v * Math.sqrt(3)).toFixed(2)})`, (v) => {
      this.uLattice = v;
      this.updateFlowInfo();
    });
    bindSlider('smag', (v) => v.toFixed(2), (v) => (this.smagorinsky = v));
    bindSlider('spf', (v) => (v ? String(v) : 'auto'), (v) => (this.stepsPerFrame = v));

    bindSlider('prad', (v) => `${v} cellules`, (v) => (this.pertRadius = v));
    bindSlider('pstr', (v) => `× ${v.toFixed(1)}`, (v) => (this.pertStrength = v));
    const setTool = (t: 'camera' | 'blow') => {
      this.tool = t;
      $('tool-camera').setAttribute('aria-pressed', String(t === 'camera'));
      $('tool-blow').setAttribute('aria-pressed', String(t === 'blow'));
      $('view').classList.toggle('blow', t === 'blow');
    };
    $('tool-camera').addEventListener('click', () => setTool('camera'));
    $('tool-blow').addEventListener('click', () => setTool('blow'));

    // slice
    const field = $<HTMLSelectElement>('slice-field');
    field.innerHTML = SLICE_FIELDS.map((f) => `<option value="${f.id}">${f.label}</option>`).join('');
    const applyRange = () => {
      const f = SLICE_FIELDS[this.slice.field];
      this.slice.min = f.range[0];
      this.slice.max = f.range[1];
      $<HTMLInputElement>('slice-min').value = String(f.range[0]);
      $<HTMLInputElement>('slice-max').value = String(f.range[1]);
      this.updateLegend();
    };
    field.addEventListener('change', () => {
      this.slice.field = +field.value;
      applyRange();
    });
    applyRange();
    $<HTMLInputElement>('slice-show').addEventListener('change', (e) => (this.slice.show = (e.target as HTMLInputElement).checked));
    document.querySelectorAll<HTMLButtonElement>('.segmented button').forEach((b) =>
      b.addEventListener('click', () => this.setSliceAxis(+b.dataset.axis!)),
    );
    this.setSliceAxis(1);
    bindSlider('slice-pos', (v) => `${Math.round(v * 100)} %`, (v) => (this.slice.pos = v));
    for (const [id, key] of [['slice-min', 'min'], ['slice-max', 'max']] as const) {
      $<HTMLInputElement>(id).addEventListener('input', (e) => {
        const v = parseFloat((e.target as HTMLInputElement).value);
        if (Number.isFinite(v)) {
          this.slice[key] = v;
          this.updateLegend();
        }
      });
    }

    // smoke
    $<HTMLInputElement>('smoke-show').addEventListener('change', (e) => (this.smoke.show = (e.target as HTMLInputElement).checked));
    $<HTMLSelectElement>('smoke-mode').addEventListener('change', (e) => (this.smoke.mode = (e.target as HTMLSelectElement).value as RakeMode));
    const pct = (v: number) => `${Math.round(v * 100)} %`;
    bindSlider('smoke-z', pct, (v) => (this.smoke.z = v));
    bindSlider('smoke-y', pct, (v) => (this.smoke.y = v));
    bindSlider('smoke-w', pct, (v) => (this.smoke.w = v));
    bindSlider('smoke-n', String, (v) => (this.smoke.nozzles = v));
    bindSlider('smoke-size', (v) => `${v} px`, (v) => (this.smoke.size = v));
    $<HTMLSelectElement>('smoke-count').addEventListener('change', (e) => (this.smoke.count = +(e.target as HTMLSelectElement).value));

    // volume
    const volMode = $<HTMLSelectElement>('vol-mode');
    const volLabel = () => {
      const t = this.volumeThreshold();
      return this.volMode === 2 ? `|ω|·L/U > ${t.toPrecision(2)}` : `Q·(L/U)² > ${t.toPrecision(2)}`;
    };
    const thr = bindSlider('vol-thr', volLabel, (v) => (this.volThr = v));
    volMode.addEventListener('change', () => {
      this.volMode = +volMode.value as 0 | 1 | 2;
      setSlider(thr, this.volThr);
    });
    $<HTMLInputElement>('surf-cp').addEventListener('change', (e) => (this.surfaceCp = (e.target as HTMLInputElement).checked));

    // measures
    $<HTMLSelectElement>('avg-window').addEventListener('change', (e) => (this.avgWindow = +(e.target as HTMLSelectElement).value));
    $<HTMLSelectElement>('ref-area').addEventListener('change', (e) => {
      this.refArea = (e.target as HTMLSelectElement).value as 'frontal' | 'planform';
      this.stats.clear();
    });
    $('btn-clear-stats').addEventListener('click', () => this.stats.clear());

    // top bar
    $('btn-pause').addEventListener('click', () => this.togglePause());
    $('btn-reset').addEventListener('click', () => this.resetFlow());
    $('btn-gust').addEventListener('click', () => this.gust());
  }

  /** Pushes model / flow state into the controls (after loading a preset or a file). */
  private syncModelControls(): void {
    setSlider($('yaw'), this.yaw);
    setSlider($('pitch'), this.pitch);
    setSlider($('roll'), this.roll);
    setSlider($('clearance'), this.clearance);
    $<HTMLSelectElement>('ground').value = this.ground;
    $('clearance-row').style.display = this.ground === 'none' ? 'none' : '';
    $<HTMLInputElement>('speed').value = String(this.speed);
    $<HTMLInputElement>('length').value = String(this.length);
    $<HTMLSelectElement>('ref-area').value = this.refArea;
  }

  // ---------------------------------------------------------------- feedback

  private toastTimer = 0;
  private toast(msg: string, kind: 'warn' | 'info' = 'warn'): void {
    const t = $('toast');
    t.textContent = msg;
    t.className = 'toast' + (kind === 'info' ? ' info' : '');
    t.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => (t.hidden = true), kind === 'info' ? 7000 : 12000);
    t.onclick = () => (t.hidden = true);
  }

  private setBusy(text: string | null): void {
    $('busy').hidden = text === null;
    if (text) $('busy-text').textContent = text;
  }
}

// ------------------------------------------------------------------ boot

function fatal(html: string): void {
  const f = $('fatal');
  f.innerHTML = html;
  f.hidden = false;
}

initGpu((msg) => fatal(`Le GPU a été perdu (${msg}).<br>Rechargez la page, éventuellement avec une résolution plus faible.`))
  .then((gpu) => {
    new App(gpu);
  })
  .catch((e: Error) => fatal(e.message));
