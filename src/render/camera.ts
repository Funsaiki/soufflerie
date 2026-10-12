import { V3, add, cross, invert, lookAt, multiply, normalize, perspective, scale, sub, transformPoint } from './math';

export interface CameraMatrices {
  viewProj: Float32Array;
  invViewProj: Float32Array;
  eye: V3;
  forward: V3;
  right: V3;
  up: V3;
}

/** Orbit camera around a target, z is up. */
export class OrbitCamera {
  target: V3 = [0, 0, 0];
  yaw = -2.35;
  pitch = 0.38;
  distance = 100;
  fov = (40 * Math.PI) / 180;

  eye(): V3 {
    const cp = Math.cos(this.pitch);
    return add(this.target, [
      this.distance * cp * Math.cos(this.yaw),
      this.distance * cp * Math.sin(this.yaw),
      this.distance * Math.sin(this.pitch),
    ]);
  }

  matrices(aspect: number): CameraMatrices {
    const eye = this.eye();
    const view = lookAt(eye, this.target, [0, 0, 1]);
    const proj = perspective(this.fov, aspect, this.distance * 0.01, this.distance * 30);
    const viewProj = multiply(proj, view);
    const forward = normalize(sub(this.target, eye));
    const right = normalize(cross(forward, [0, 0, 1]));
    const up = cross(right, forward);
    return { viewProj, invViewProj: invert(viewProj), eye, forward, right, up };
  }

  orbit(dx: number, dy: number): void {
    this.yaw -= dx * 0.006;
    this.pitch = Math.min(1.55, Math.max(-1.55, this.pitch + dy * 0.006));
  }

  pan(dx: number, dy: number, viewportHeight: number, m: CameraMatrices): void {
    const s = (2 * this.distance * Math.tan(this.fov / 2)) / viewportHeight;
    this.target = add(this.target, add(scale(m.right, -dx * s), scale(m.up, dy * s)));
  }

  zoom(factor: number): void {
    this.distance = Math.min(1e5, Math.max(1, this.distance * factor));
  }

  fit(center: V3, radius: number): void {
    this.target = center;
    this.distance = (radius / Math.sin(this.fov / 2)) * 1.05;
  }

  /** World-space ray through a pixel given in normalized device coordinates. */
  ray(m: CameraMatrices, ndcX: number, ndcY: number): { origin: V3; dir: V3 } {
    const near = transformPoint(m.invViewProj, ndcX, ndcY, 0);
    const far = transformPoint(m.invViewProj, ndcX, ndcY, 1);
    return { origin: near, dir: normalize(sub(far, near)) };
  }
}
