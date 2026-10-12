export interface Sample {
  step: number;
  cd: number;
  cl: number;
  cs: number;
}

export interface WindowStats {
  cd: number;
  cl: number;
  cs: number;
  cdStd: number;
  clStd: number;
  /** Strouhal number from the dominant transverse oscillation, NaN if none is detected. */
  strouhal: number;
  samples: number;
}

const MAX_SAMPLES = 6000;

export class Measurements {
  samples: Sample[] = [];

  add(s: Sample): void {
    this.samples.push(s);
    if (this.samples.length > MAX_SAMPLES) this.samples.splice(0, this.samples.length - MAX_SAMPLES);
  }

  clear(): void {
    this.samples = [];
  }

  /** Statistics over the last `steps` lattice steps. */
  window(steps: number, lengthCells: number, U: number): WindowStats {
    const all = this.samples;
    const empty = { cd: NaN, cl: NaN, cs: NaN, cdStd: NaN, clStd: NaN, strouhal: NaN, samples: 0 };
    if (!all.length) return empty;
    const last = all[all.length - 1].step;
    let i0 = all.length - 1;
    while (i0 > 0 && all[i0 - 1].step >= last - steps) i0--;
    const w = all.slice(i0);
    if (w.length < 2) return { ...empty, cd: w[0].cd, cl: w[0].cl, cs: w[0].cs, samples: 1 };
    const mean = (k: 'cd' | 'cl' | 'cs') => w.reduce((a, s) => a + s[k], 0) / w.length;
    const std = (k: 'cd' | 'cl' | 'cs', m: number) => Math.sqrt(w.reduce((a, s) => a + (s[k] - m) ** 2, 0) / w.length);
    const cd = mean('cd'), cl = mean('cl'), cs = mean('cs');
    const clStd = std('cl', cl), csStd = std('cs', cs);
    const key = clStd >= csStd ? 'cl' : 'cs';
    const m = key === 'cl' ? cl : cs;
    const sd = Math.max(clStd, csStd);
    // mean crossings with hysteresis
    let crossings = 0, first = -1, lastCross = -1, state = 0;
    for (const s of w) {
      const v = s[key] - m;
      const ns = v > 0.25 * sd ? 1 : v < -0.25 * sd ? -1 : state;
      if (state !== 0 && ns !== state) {
        crossings++;
        if (first < 0) first = s.step;
        lastCross = s.step;
      }
      state = ns;
    }
    let strouhal = NaN;
    if (crossings >= 5 && sd > 1e-3 && lastCross > first) {
      const freq = (crossings - 1) / 2 / (lastCross - first);
      strouhal = (freq * lengthCells) / U;
    }
    return { cd, cl, cs, cdStd: std('cd', cd), clStd, strouhal, samples: w.length };
  }
}
