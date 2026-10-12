export interface Gpu {
  device: GPUDevice;
  format: GPUTextureFormat;
  /** Largest lattice the device can hold (one distribution buffer per binding). */
  maxCells: number;
  name: string;
}

export async function initGpu(onLost: (msg: string) => void): Promise<Gpu> {
  if (!('gpu' in navigator)) {
    throw new Error(
      "WebGPU n'est pas disponible dans ce navigateur.<br>Utilisez une version récente de Chrome, Edge ou Opera (ou Firefox ≥ 141 / Safari ≥ 26), avec l'accélération matérielle activée.",
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error("Aucune carte graphique compatible WebGPU n'a été trouvée.");
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
      maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
    },
  });
  device.addEventListener('uncapturederror', (e) => console.error('[WebGPU]', (e as GPUUncapturedErrorEvent).error.message));
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') onLost(info.message || 'périphérique perdu');
  });
  const perBuffer = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const maxCells = Math.floor(perBuffer / (19 * 4));
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  const name = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') : '';
  return { device, format: navigator.gpu.getPreferredCanvasFormat(), maxCells, name };
}
