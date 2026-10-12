/**
 * Conversion between the physical problem and lattice units.
 *
 * The simulation is governed by the Reynolds number Re = U L / nu. The lattice inflow speed
 * u_lb is fixed (small, to keep the lattice Mach number low); the lattice viscosity follows:
 *   nu_lb = u_lb * L_cells / Re,  tau = 3 nu_lb + 1/2
 * Physical scales: dx = L / L_cells, dt = u_lb * dx / U.
 */

export const FLUIDS = {
  air: { label: 'Air (20 °C)', nu: 1.516e-5, rho: 1.204 },
  water: { label: 'Eau (20 °C)', nu: 1.004e-6, rho: 998.2 },
} as const;

export type FluidId = keyof typeof FLUIDS;

export interface FlowSetup {
  fluid: FluidId;
  /** Free-stream speed, m/s. */
  speed: number;
  /** Body length along the flow, m. */
  length: number;
  /** Lattice inflow speed (cells / step). */
  uLattice: number;
  /** Body length along the flow, cells. */
  lengthCells: number;
}

export interface FlowScales {
  reynolds: number;
  nuLattice: number;
  tau: number;
  mach: number;
  dx: number;
  dt: number;
  /** Resolved Reynolds number per cell: above ~ 1 the wall layers are not resolved and LES takes over. */
  cellReynolds: number;
}

export function flowScales(s: FlowSetup): FlowScales {
  const fluid = FLUIDS[s.fluid];
  const reynolds = (s.speed * s.length) / fluid.nu;
  const nuLattice = (s.uLattice * s.lengthCells) / reynolds;
  const dx = s.length / s.lengthCells;
  return {
    reynolds,
    nuLattice,
    tau: 3 * nuLattice + 0.5,
    mach: s.uLattice * Math.sqrt(3),
    dx,
    dt: (s.uLattice * dx) / s.speed,
    cellReynolds: s.uLattice / nuLattice,
  };
}

export function formatSI(v: number, unit: string, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  const pre: [number, string][] = [[1e9, 'G'], [1e6, 'M'], [1e3, 'k'], [1, ''], [1e-3, 'm'], [1e-6, 'µ'], [1e-9, 'n']];
  for (const [f, p] of pre) if (a >= f || f === 1e-9) return `${(v / f).toPrecision(digits)} ${p}${unit}`;
  return `${v} ${unit}`;
}

export function formatReynolds(re: number): string {
  if (re >= 1e4) {
    const e = Math.floor(Math.log10(re));
    return `${(re / 10 ** e).toFixed(2)}·10${superscript(e)}`;
  }
  return re.toFixed(re < 10 ? 2 : 0);
}

function superscript(n: number): string {
  const map = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  return String(n).split('').map((d) => map[+d]).join('');
}
