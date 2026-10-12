import type { Mesh } from './mesh';

export type UpAxis = 'z' | 'y';

export interface LoadedModel {
  mesh: Mesh;
  /** Up axis conventionally used by the format (the user can override it). */
  upAxis: UpAxis;
}

export const ACCEPTED_EXTENSIONS = '.stl,.obj,.glb,.gltf';

export async function loadModelFile(file: File): Promise<LoadedModel> {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  const buf = await file.arrayBuffer();
  let positions: Float32Array;
  let upAxis: UpAxis;
  switch (ext) {
    case 'stl':
      positions = parseSTL(buf);
      upAxis = 'z';
      break;
    case 'obj':
      positions = parseOBJ(new TextDecoder().decode(buf));
      upAxis = 'y';
      break;
    case 'glb':
    case 'gltf':
      positions = parseGLTF(buf, ext === 'glb');
      upAxis = 'y';
      break;
    default:
      throw new Error(`Format « .${ext} » non pris en charge. Formats acceptés : STL, OBJ, glTF/GLB.`);
  }
  if (positions.length < 9) throw new Error('Le fichier ne contient aucun triangle.');
  for (let i = 0; i < positions.length; i++) {
    if (!Number.isFinite(positions[i])) throw new Error('Le fichier contient des coordonnées invalides.');
  }
  return { mesh: { name: file.name, positions }, upAxis };
}

// ---------------------------------------------------------------- STL

export function parseSTL(buf: ArrayBuffer): Float32Array {
  const dv = new DataView(buf);
  if (buf.byteLength >= 84) {
    const count = dv.getUint32(80, true);
    if (84 + count * 50 === buf.byteLength) return parseBinarySTL(dv, count);
  }
  const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 1024)));
  if (/^\s*solid/i.test(head)) {
    const text = new TextDecoder().decode(buf);
    if (/facet/i.test(text)) return parseAsciiSTL(text);
  }
  if (buf.byteLength >= 84) {
    const count = dv.getUint32(80, true);
    if (84 + count * 50 <= buf.byteLength) return parseBinarySTL(dv, count);
  }
  throw new Error('Fichier STL illisible.');
}

function parseBinarySTL(dv: DataView, count: number): Float32Array {
  const out = new Float32Array(count * 9);
  for (let t = 0; t < count; t++) {
    const o = 84 + t * 50 + 12; // skip the stored normal
    for (let k = 0; k < 9; k++) out[t * 9 + k] = dv.getFloat32(o + k * 4, true);
  }
  return out;
}

function parseAsciiSTL(text: string): Float32Array {
  const re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/gi;
  const values: number[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) values.push(+m[1], +m[2], +m[3]);
  return new Float32Array(values.slice(0, values.length - (values.length % 9)));
}

// ---------------------------------------------------------------- OBJ

export function parseOBJ(text: string): Float32Array {
  const verts: number[] = [];
  const tris: number[] = [];
  const lines = text.split(/\r?\n/);
  for (let line of lines) {
    line = line.trim();
    if (line.startsWith('v ')) {
      const p = line.split(/\s+/);
      verts.push(+p[1], +p[2], +p[3]);
    } else if (line.startsWith('f ')) {
      const p = line.split(/\s+/).slice(1);
      const idx = p.map((tok) => {
        const i = parseInt(tok.split('/')[0], 10);
        return i < 0 ? verts.length / 3 + i : i - 1;
      });
      for (let k = 1; k + 1 < idx.length; k++) {
        for (const i of [idx[0], idx[k], idx[k + 1]]) {
          if (i < 0 || i * 3 + 2 >= verts.length) throw new Error('Fichier OBJ : indice de sommet invalide.');
          tris.push(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
        }
      }
    }
  }
  return new Float32Array(tris);
}

// ---------------------------------------------------------------- glTF / GLB

interface GltfJson {
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: {
    children?: number[];
    mesh?: number;
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
  }[];
  meshes?: { primitives: { attributes: Record<string, number>; indices?: number; mode?: number; extensions?: Record<string, unknown> }[] }[];
  accessors?: { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean }[];
  bufferViews?: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
  buffers?: { uri?: string; byteLength: number }[];
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

export function parseGLTF(buf: ArrayBuffer, isGlb: boolean): Float32Array {
  let json: GltfJson;
  const buffers: ArrayBuffer[] = [];
  if (isGlb) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('Fichier GLB invalide.');
    let offset = 12;
    let jsonText = '';
    let bin: ArrayBuffer | null = null;
    while (offset + 8 <= buf.byteLength) {
      const len = dv.getUint32(offset, true);
      const type = dv.getUint32(offset + 4, true);
      const chunk = buf.slice(offset + 8, offset + 8 + len);
      if (type === 0x4e4f534a) jsonText = new TextDecoder().decode(chunk);
      else if (type === 0x004e4942) bin = chunk;
      offset += 8 + len;
    }
    json = JSON.parse(jsonText);
    (json.buffers ?? []).forEach((b, i) => {
      if (i === 0 && !b.uri && bin) buffers.push(bin);
      else buffers.push(decodeDataUri(b.uri));
    });
  } else {
    json = JSON.parse(new TextDecoder().decode(buf));
    for (const b of json.buffers ?? []) buffers.push(decodeDataUri(b.uri));
  }

  const out: number[] = [];

  const readAccessor = (index: number): { data: number[]; comps: number } => {
    const acc = json.accessors![index];
    const comps = COMPONENTS[acc.type];
    const data = new Array<number>(acc.count * comps).fill(0);
    if (acc.bufferView === undefined) return { data, comps };
    const view = json.bufferViews![acc.bufferView];
    const dv = new DataView(buffers[view.buffer]);
    const size = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }[acc.componentType] ?? 4;
    const stride = view.byteStride || size * comps;
    const base = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    for (let i = 0; i < acc.count; i++) {
      for (let c = 0; c < comps; c++) {
        const o = base + i * stride + c * size;
        let v: number;
        switch (acc.componentType) {
          case 5126: v = dv.getFloat32(o, true); break;
          case 5125: v = dv.getUint32(o, true); break;
          case 5123: v = dv.getUint16(o, true); break;
          case 5121: v = dv.getUint8(o); break;
          case 5122: v = dv.getInt16(o, true); break;
          case 5120: v = dv.getInt8(o); break;
          default: throw new Error('glTF : type de composante inconnu.');
        }
        data[i * comps + c] = v;
      }
    }
    return { data, comps };
  };

  const visit = (nodeIndex: number, parent: number[]) => {
    const node = json.nodes![nodeIndex];
    const world = mat4Mul(parent, nodeMatrix(node));
    if (node.mesh !== undefined) {
      for (const prim of json.meshes![node.mesh].primitives) {
        if (prim.extensions && 'KHR_draco_mesh_compression' in prim.extensions) {
          throw new Error('glTF compressé Draco non pris en charge : exportez sans compression.');
        }
        if ((prim.mode ?? 4) !== 4 || prim.attributes.POSITION === undefined) continue;
        const pos = readAccessor(prim.attributes.POSITION).data;
        const count = pos.length / 3;
        const idx = prim.indices !== undefined ? readAccessor(prim.indices).data : [...Array(count).keys()];
        for (let k = 0; k + 2 < idx.length; k += 3) {
          for (let v = 0; v < 3; v++) {
            const i = idx[k + v] * 3;
            const x = pos[i], y = pos[i + 1], z = pos[i + 2];
            out.push(
              world[0] * x + world[4] * y + world[8] * z + world[12],
              world[1] * x + world[5] * y + world[9] * z + world[13],
              world[2] * x + world[6] * y + world[10] * z + world[14],
            );
          }
        }
      }
    }
    for (const c of node.children ?? []) visit(c, world);
  };

  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const scene = json.scenes?.[json.scene ?? 0];
  const roots = scene?.nodes ?? (json.nodes ?? []).map((_, i) => i);
  for (const r of roots) visit(r, identity);
  return new Float32Array(out);
}

function decodeDataUri(uri: string | undefined): ArrayBuffer {
  if (!uri || !uri.startsWith('data:')) {
    throw new Error('glTF avec fichier .bin externe non pris en charge : utilisez le format .glb (binaire).');
  }
  const b64 = uri.slice(uri.indexOf(',') + 1);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function nodeMatrix(node: NonNullable<GltfJson['nodes']>[number]): number[] {
  if (node.matrix) return node.matrix;
  const [tx, ty, tz] = node.translation ?? [0, 0, 0];
  const [qx, qy, qz, qw] = node.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale ?? [1, 1, 1];
  return [
    (1 - 2 * (qy * qy + qz * qz)) * sx, 2 * (qx * qy + qz * qw) * sx, 2 * (qx * qz - qy * qw) * sx, 0,
    2 * (qx * qy - qz * qw) * sy, (1 - 2 * (qx * qx + qz * qz)) * sy, 2 * (qy * qz + qx * qw) * sy, 0,
    2 * (qx * qz + qy * qw) * sz, 2 * (qy * qz - qx * qw) * sz, (1 - 2 * (qx * qx + qy * qy)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function mat4Mul(a: number[], b: number[]): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}
