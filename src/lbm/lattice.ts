/** D3Q19 velocity set. Opposite directions are adjacent pairs (1,2), (3,4), ... */
export const C: [number, number, number][] = [
  [0, 0, 0],
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
  [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1],
];

export const Q = C.length;

export const W = C.map((c, i) => (i === 0 ? 1 / 3 : Math.abs(c[0]) + Math.abs(c[1]) + Math.abs(c[2]) === 1 ? 1 / 18 : 1 / 36));

export const OPP = C.map((c) => C.findIndex((d) => d[0] === -c[0] && d[1] === -c[1] && d[2] === -c[2]));
