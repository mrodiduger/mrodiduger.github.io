const MOBILE_BREAKPOINT = 720;
const MAX_PIXEL_RATIO = 2;

export function getSimulationProfile(viewportWidth) {
  return viewportWidth <= MOBILE_BREAKPOINT
    ? { particleCount: 384, cssHeight: 168 }
    : { particleCount: 600, cssHeight: 240 };
}

export function getCanvasMetrics(cssWidth, cssHeight, devicePixelRatio = 1) {
  const pixelRatio = Math.min(Math.max(devicePixelRatio, 1), MAX_PIXEL_RATIO);
  return {
    width: Math.max(1, Math.round(cssWidth * pixelRatio)),
    height: Math.max(1, Math.round(cssHeight * pixelRatio)),
    pixelRatio,
  };
}

export function clientPointToSimulation(clientX, clientY, rect) {
  const aspect = rect.width / rect.height;
  return {
    x: ((clientX - rect.left) / rect.width) * aspect,
    y: 1 - (clientY - rect.top) / rect.height,
  };
}

export function shouldAnimate(state) {
  return state.inViewport && state.pageVisible && !state.reducedMotion && !state.failed && !state.idle;
}

export function consumeActiveFrameTime(remainingMs, previousFrameTime, frameTime) {
  const elapsedMs = previousFrameTime === null
    ? 0
    : Math.max(0, frameTime - previousFrameTime);
  return {
    remainingMs: Math.max(0, remainingMs - elapsedMs),
    previousFrameTime: frameTime,
  };
}

export function createInitialParticles(count, bounds, settled = false, random = Math.random) {
  const data = new Float32Array(count * 4);
  const blockWidth = bounds.x * (settled ? 0.92 : 0.46);
  const blockHeight = settled ? 0.24 : 0.72;
  const columns = Math.ceil(Math.sqrt((count * blockWidth) / blockHeight));
  const rows = Math.ceil(count / columns);
  const spacing = Math.min(blockWidth / columns, blockHeight / rows);
  const startX = settled ? (bounds.x - blockWidth) / 2 : bounds.x * 0.07;
  const startY = 0.055;

  for (let index = 0; index < count; index += 1) {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const jitter = (random() - 0.5) * spacing * 0.12;
    const offset = index * 4;
    data[offset] = startX + (column + 0.5) * spacing + jitter;
    data[offset + 1] = startY + (row + 0.5) * spacing;
    data[offset + 2] = 0;
    data[offset + 3] = 0;
  }

  return data;
}

export const COMPUTE_SHADER = /* wgsl */ `
struct Particle {
  position: vec2f,
  velocity: vec2f,
}

struct SimUniforms {
  bounds: vec2f,
  pointer: vec2f,
  particleCount: u32,
  pointerActive: u32,
  deltaTime: f32,
  smoothingRadius: f32,
  particleRadius: f32,
  mass: f32,
  restDensity: f32,
  gasConstant: f32,
  viscosity: f32,
  gravity: f32,
  pointerRadius: f32,
  pointerStrength: f32,
}

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> scalarField: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vectorField: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(4) var<uniform> uniforms: SimUniforms;

const PI = 3.14159265359;

fn poly6(distanceSquared: f32, h: f32) -> f32 {
  let difference = h * h - distanceSquared;
  return select(0.0, (4.0 / (PI * pow(h, 8.0))) * difference * difference * difference,
    difference > 0.0);
}

fn spikyGradient(delta: vec2f, distance: f32, h: f32) -> vec2f {
  if (distance <= 0.0001 || distance >= h) { return vec2f(0.0); }
  let scale = (-30.0 / (PI * pow(h, 5.0))) * (h - distance) * (h - distance);
  return scale * delta / distance;
}

fn viscosityLaplacian(distance: f32, h: f32) -> f32 {
  return select(0.0, (20.0 / (3.0 * PI * pow(h, 5.0))) * (h - distance), distance < h);
}

@compute @workgroup_size(64)
fn densityPressure(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x;
  if (index >= uniforms.particleCount) { return; }
  var density = 0.0;
  for (var other = 0u; other < uniforms.particleCount; other += 1u) {
    let delta = particlesIn[index].position - particlesIn[other].position;
    density += uniforms.mass * poly6(dot(delta, delta), uniforms.smoothingRadius);
  }
  density = max(density, uniforms.restDensity * 0.25);
  scalarField[index] = vec2f(density,
    uniforms.gasConstant * (density - uniforms.restDensity));
}

@compute @workgroup_size(64)
fn forces(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x;
  if (index >= uniforms.particleCount) { return; }
  let particle = particlesIn[index];
  let density = scalarField[index].x;
  let pressure = scalarField[index].y;
  var acceleration = vec2f(0.0, -uniforms.gravity);

  for (var other = 0u; other < uniforms.particleCount; other += 1u) {
    if (other == index) { continue; }
    let neighbor = particlesIn[other];
    let delta = particle.position - neighbor.position;
    let distance = length(delta);
    if (distance < uniforms.smoothingRadius) {
      let neighborDensity = scalarField[other].x;
      let neighborPressure = scalarField[other].y;
      acceleration -= uniforms.mass *
        (pressure / (density * density) + neighborPressure /
        (neighborDensity * neighborDensity)) *
        spikyGradient(delta, distance, uniforms.smoothingRadius);
      acceleration += uniforms.viscosity * uniforms.mass *
        (neighbor.velocity - particle.velocity) / neighborDensity *
        viscosityLaplacian(distance, uniforms.smoothingRadius);
    }
  }

  if (uniforms.pointerActive == 1u) {
    let pointerDelta = particle.position - uniforms.pointer;
    let pointerDistance = length(pointerDelta);
    if (pointerDistance > 0.0001 && pointerDistance < uniforms.pointerRadius) {
      let falloff = 1.0 - pointerDistance / uniforms.pointerRadius;
      acceleration += normalize(pointerDelta) * uniforms.pointerStrength * falloff * falloff;
    }
  }
  vectorField[index] = acceleration;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x;
  if (index >= uniforms.particleCount) { return; }
  var velocity = (particlesIn[index].velocity + vectorField[index] * uniforms.deltaTime) * 0.998;
  var position = particlesIn[index].position + velocity * uniforms.deltaTime;
  let radius = uniforms.particleRadius;

  if (position.x < radius) { position.x = radius; velocity.x = abs(velocity.x) * 0.35; }
  if (position.x > uniforms.bounds.x - radius) {
    position.x = uniforms.bounds.x - radius; velocity.x = -abs(velocity.x) * 0.35;
  }
  if (position.y < radius) { position.y = radius; velocity.y = abs(velocity.y) * 0.28; }
  if (position.y > uniforms.bounds.y - radius) {
    position.y = uniforms.bounds.y - radius; velocity.y = -abs(velocity.y) * 0.28;
  }
  particlesOut[index].position = position;
  particlesOut[index].velocity = velocity;
}
`;

export const RENDER_SHADER = /* wgsl */ `
struct Particle {
  position: vec2f,
  velocity: vec2f,
}

struct RenderUniforms {
  bounds: vec2f,
  particleRadius: f32,
  surfaceRadius: f32,
  ink: vec4f,
  accent: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) color: vec4f,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
  );
  let local = corners[vertexIndex];
  let world = particles[instanceIndex].position + local * uniforms.particleRadius;
  let accentParticle = instanceIndex % 19u == 0u;
  let paletteColor = select(uniforms.ink, uniforms.accent, accentParticle);
  let alpha = select(0.72, 0.92, accentParticle);

  var output: VertexOutput;
  output.position = vec4f(world / uniforms.bounds * 2.0 - vec2f(1.0), 0.0, 1.0);
  output.local = local;
  output.color = vec4f(paletteColor.rgb * alpha, alpha);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  if (length(input.local) > 1.0) { discard; }
  return input.color;
}
`;

export const DENSITY_RENDER_SHADER = /* wgsl */ `
struct Particle {
  position: vec2f,
  velocity: vec2f,
}

struct RenderUniforms {
  bounds: vec2f,
  particleRadius: f32,
  surfaceRadius: f32,
  ink: vec4f,
  accent: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

@vertex
fn densityVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
  );
  let local = corners[vertexIndex];
  let world = particles[instanceIndex].position + local * uniforms.surfaceRadius;

  var output: VertexOutput;
  output.position = vec4f(world / uniforms.bounds * 2.0 - vec2f(1.0), 0.0, 1.0);
  output.local = local;
  return output;
}

@fragment
fn densityFragment(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.local);
  let density = pow(max(1.0 - distance, 0.0), 2.0);
  return vec4f(density, 0.0, 0.0, density);
}
`;

export const COMPOSITE_SHADER = /* wgsl */ `
struct RenderUniforms {
  bounds: vec2f,
  particleRadius: f32,
  surfaceRadius: f32,
  ink: vec4f,
  accent: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var densitySampler: sampler;
@group(0) @binding(1) var densityTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: RenderUniforms;

@vertex
fn compositeVertex(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  let position = positions[vertexIndex];

  var output: VertexOutput;
  output.position = vec4f(position, 0.0, 1.0);
  output.uv = position * vec2f(0.5, -0.5) + vec2f(0.5);
  return output;
}

@fragment
fn compositeFragment(input: VertexOutput) -> @location(0) vec4f {
  let density = textureSample(densityTexture, densitySampler, input.uv).r;
  let alpha = smoothstep(0.42, 0.62, density);
  return vec4f(uniforms.ink.rgb * alpha, alpha);
}
`;

export const SIMULATION_UNIFORM_BYTES = 64;
const RENDER_UNIFORM_BYTES = 48;
const PARTICLE_BYTES = 16;
const VECTOR_BYTES = 8;
const WORKGROUP_SIZE = 64;
const DENSITY_TEXTURE_FORMAT = "rgba8unorm";

const DELTA_TIME = 1 / 120;
const REST_DENSITY = 1;
const DEFAULT_PRESSURE_MULTIPLIER = 0.55;
const VISCOSITY = 0.08;
const GRAVITY = 1.35;
const POINTER_RADIUS = 0.22;
const DEFAULT_POINTER_STRENGTH = 5.0;

function hexToRgba(value) {
  const hex = value.trim().slice(1);
  return [
    Number.parseInt(hex.slice(0, 2), 16) / 255,
    Number.parseInt(hex.slice(2, 4), 16) / 255,
    Number.parseInt(hex.slice(4, 6), 16) / 255,
    1,
  ];
}

export class SphSimulation {
  static async create(canvas, profile, options = { settled: false }) {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error("WebGPU is unavailable");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No WebGPU adapter is available");
    }

    const device = await adapter.requestDevice();
    if (!device) {
      throw new Error("No WebGPU device is available");
    }

    const buffers = [];
    let context = null;
    let contextConfigured = false;
    let errorScopeOpen = false;

    try {
      context = canvas?.getContext?.("webgpu");
      if (!context) {
        throw new Error("No WebGPU canvas context is available");
      }

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "premultiplied" });
      contextConfigured = true;

      const width = Math.max(1, canvas.width);
      const height = Math.max(1, canvas.height);
      const bounds = { x: width / height, y: 1 };
      const particleCount = profile.particleCount;
      const particleData = createInitialParticles(
        particleCount,
        bounds,
        options.settled ?? false,
      );
      device.pushErrorScope("validation");
      errorScopeOpen = true;
      const createBuffer = (size, usage) => {
        const buffer = device.createBuffer({ size, usage });
        buffers.push(buffer);
        return buffer;
      };
      const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const uniformUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
      const particleBuffers = [
        createBuffer(particleCount * PARTICLE_BYTES, storageUsage),
        createBuffer(particleCount * PARTICLE_BYTES, storageUsage),
      ];
      const scalarBuffer = createBuffer(particleCount * VECTOR_BYTES, storageUsage);
      const vectorBuffer = createBuffer(particleCount * VECTOR_BYTES, storageUsage);
      const simulationUniformBuffer = createBuffer(SIMULATION_UNIFORM_BYTES, uniformUsage);
      const renderUniformBuffer = createBuffer(RENDER_UNIFORM_BYTES, uniformUsage);

      device.queue.writeBuffer(particleBuffers[0], 0, particleData);
      device.queue.writeBuffer(particleBuffers[1], 0, particleData);

      const computeBindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
          {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" },
          },
        ],
      });
      const compositeBindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        ],
      });
      const computeModule = device.createShaderModule({ code: COMPUTE_SHADER });
      const renderModule = device.createShaderModule({ code: RENDER_SHADER });
      const densityRenderModule = device.createShaderModule({ code: DENSITY_RENDER_SHADER });
      const compositeModule = device.createShaderModule({ code: COMPOSITE_SHADER });
      const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] });
      const renderLayout = device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] });
      const compositeLayout = device.createPipelineLayout({
        bindGroupLayouts: [compositeBindGroupLayout],
      });
      const [
        densityPipeline,
        forcePipeline,
        integratePipeline,
        renderPipeline,
        densityRenderPipeline,
        compositePipeline,
      ] = await Promise.all([
        device.createComputePipelineAsync({
          layout: computeLayout,
          compute: { module: computeModule, entryPoint: "densityPressure" },
        }),
        device.createComputePipelineAsync({
          layout: computeLayout,
          compute: { module: computeModule, entryPoint: "forces" },
        }),
        device.createComputePipelineAsync({
          layout: computeLayout,
          compute: { module: computeModule, entryPoint: "integrate" },
        }),
        device.createRenderPipelineAsync({
          layout: renderLayout,
          vertex: { module: renderModule, entryPoint: "vertexMain" },
          fragment: {
            module: renderModule,
            entryPoint: "fragmentMain",
            targets: [{
              format,
              blend: {
                color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              },
            }],
          },
          primitive: { topology: "triangle-list" },
        }),
        device.createRenderPipelineAsync({
          layout: renderLayout,
          vertex: { module: densityRenderModule, entryPoint: "densityVertex" },
          fragment: {
            module: densityRenderModule,
            entryPoint: "densityFragment",
            targets: [{
              format: DENSITY_TEXTURE_FORMAT,
              blend: {
                color: { srcFactor: "one", dstFactor: "one" },
                alpha: { srcFactor: "one", dstFactor: "one" },
              },
            }],
          },
          primitive: { topology: "triangle-list" },
        }),
        device.createRenderPipelineAsync({
          layout: compositeLayout,
          vertex: { module: compositeModule, entryPoint: "compositeVertex" },
          fragment: {
            module: compositeModule,
            entryPoint: "compositeFragment",
            targets: [{
              format,
              blend: {
                color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              },
            }],
          },
          primitive: { topology: "triangle-list" },
        }),
      ]);

      const computeBindGroups = particleBuffers.map((particleBuffer, index) =>
        device.createBindGroup({
          layout: computeBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: scalarBuffer } },
            { binding: 2, resource: { buffer: vectorBuffer } },
            { binding: 3, resource: { buffer: particleBuffers[1 - index] } },
            { binding: 4, resource: { buffer: simulationUniformBuffer } },
          ],
        }),
      );
      const renderBindGroups = particleBuffers.map((particleBuffer) =>
        device.createBindGroup({
          layout: renderBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: renderUniformBuffer } },
          ],
        }),
      );
      const densitySampler = device.createSampler({
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
        magFilter: "linear",
        minFilter: "linear",
      });

      errorScopeOpen = false;
      const validationResult = device.popErrorScope();
      const validationError = await validationResult;
      if (validationError) {
        throw new Error(`WebGPU validation failed: ${validationError.message}`);
      }

      const style = getComputedStyle(document.documentElement);
      const palette = {
        ink: hexToRgba(style.getPropertyValue("--ink")),
        accent: hexToRgba(style.getPropertyValue("--accent")),
      };
      const simulation = new SphSimulation({
        canvas,
        context,
        device,
        particleCount,
        particleBuffers,
        scalarBuffer,
        vectorBuffer,
        simulationUniformBuffer,
        renderUniformBuffer,
        compositeBindGroupLayout,
        densityPipeline,
        forcePipeline,
        integratePipeline,
        renderPipeline,
        densityRenderPipeline,
        compositePipeline,
        densitySampler,
        computeBindGroups,
        renderBindGroups,
        palette,
        pressureMultiplier: options.pressureMultiplier ?? DEFAULT_PRESSURE_MULTIPLIER,
        pointerStrength: options.pointerStrength ?? DEFAULT_POINTER_STRENGTH,
        renderMode: options.renderMode ?? "balls",
      });
      simulation.resize(width, height);
      return simulation;
    } catch (error) {
      if (errorScopeOpen) {
        errorScopeOpen = false;
        try {
          await device.popErrorScope();
        } catch {
          // Preserve the original initialization error.
        }
      }
      for (const buffer of buffers) {
        try {
          buffer.destroy();
        } catch {
          // Preserve the original initialization error.
        }
      }
      if (contextConfigured) {
        try {
          context.unconfigure();
        } catch {
          // Preserve the original initialization error.
        }
      }
      try {
        device.destroy();
      } catch {
        // Preserve the original initialization error.
      }
      throw error;
    }
  }

  constructor(resources) {
    Object.assign(this, resources);
    this.pressureMultiplier = resources.pressureMultiplier ?? DEFAULT_PRESSURE_MULTIPLIER;
    this.pointerStrength = resources.pointerStrength ?? DEFAULT_POINTER_STRENGTH;
    this.renderMode = resources.renderMode ?? "balls";
    this.currentParticleIndex = 0;
    this.pointer = null;
    this.destroyed = false;
  }

  resize(width, height) {
    this.canvas.width = Math.max(1, Math.round(width));
    this.canvas.height = Math.max(1, Math.round(height));
    this.bounds = { x: this.canvas.width / this.canvas.height, y: 1 };
    this.ensureDensityTarget();

    const targetPoolArea = this.bounds.x * 0.28;
    const nominalSpacing = Math.sqrt(targetPoolArea / this.particleCount);
    this.smoothingRadius = nominalSpacing * 2.15;
    this.particleRadius = nominalSpacing * 0.34;
    this.mass = targetPoolArea / this.particleCount;
    this.writeRenderUniforms();
  }

  ensureDensityTarget() {
    const width = Math.max(1, Math.ceil(this.canvas.width / 2));
    const height = Math.max(1, Math.ceil(this.canvas.height / 2));
    if (
      this.densityTexture &&
      this.densityTextureWidth === width &&
      this.densityTextureHeight === height
    ) {
      return;
    }

    const previousTexture = this.densityTexture;
    const densityTexture = this.device.createTexture({
      size: [width, height],
      format: DENSITY_TEXTURE_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const densityTextureView = densityTexture.createView();
    const compositeBindGroup = this.device.createBindGroup({
      layout: this.compositeBindGroupLayout,
      entries: [
        { binding: 0, resource: this.densitySampler },
        { binding: 1, resource: densityTextureView },
        { binding: 2, resource: { buffer: this.renderUniformBuffer } },
      ],
    });

    this.densityTexture = densityTexture;
    this.densityTextureView = densityTextureView;
    this.compositeBindGroup = compositeBindGroup;
    this.densityTextureWidth = width;
    this.densityTextureHeight = height;
    previousTexture?.destroy();
  }

  setRenderMode(mode) {
    if (this.destroyed) {
      throw new Error("Cannot update a destroyed SPH simulation");
    }
    this.renderMode = mode;
  }

  setPointer(pointOrNull) {
    this.pointer = pointOrNull;
  }

  setParameters({
    pressureMultiplier = this.pressureMultiplier,
    pointerStrength = this.pointerStrength,
  } = {}) {
    if (this.destroyed) {
      throw new Error("Cannot update a destroyed SPH simulation");
    }
    this.pressureMultiplier = pressureMultiplier;
    this.pointerStrength = pointerStrength;
    this.writeSimulationUniforms();
  }

  reset({ settled = false } = {}) {
    if (this.destroyed) {
      throw new Error("Cannot reset a destroyed SPH simulation");
    }
    const particleData = createInitialParticles(
      this.particleCount,
      this.bounds,
      settled,
    );
    this.device.queue.writeBuffer(this.particleBuffers[0], 0, particleData);
    this.device.queue.writeBuffer(this.particleBuffers[1], 0, particleData);
    this.currentParticleIndex = 0;
    this.pointer = null;
    this.writeSimulationUniforms();
    this.writeRenderUniforms();
  }

  advance(substeps = 2) {
    this.writeSimulationUniforms();
    const encoder = this.device.createCommandEncoder();
    const workgroups = Math.ceil(this.particleCount / WORKGROUP_SIZE);

    for (let step = 0; step < substeps; step += 1) {
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, this.computeBindGroups[this.currentParticleIndex]);
      pass.setPipeline(this.densityPipeline);
      pass.dispatchWorkgroups(workgroups);
      pass.setPipeline(this.forcePipeline);
      pass.dispatchWorkgroups(workgroups);
      pass.setPipeline(this.integratePipeline);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      this.currentParticleIndex = 1 - this.currentParticleIndex;
    }

    this.device.queue.submit([encoder.finish()]);
  }

  render() {
    const encoder = this.device.createCommandEncoder();
    const canvasAttachment = {
      view: this.context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    };

    if (this.renderMode === "balls") {
      const pass = encoder.beginRenderPass({ colorAttachments: [canvasAttachment] });
      pass.setPipeline(this.renderPipeline);
      pass.setBindGroup(0, this.renderBindGroups[this.currentParticleIndex]);
      pass.draw(6, this.particleCount);
      pass.end();
    } else {
      const densityPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.densityTextureView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      densityPass.setPipeline(this.densityRenderPipeline);
      densityPass.setBindGroup(0, this.renderBindGroups[this.currentParticleIndex]);
      densityPass.draw(6, this.particleCount);
      densityPass.end();

      const compositePass = encoder.beginRenderPass({ colorAttachments: [canvasAttachment] });
      compositePass.setPipeline(this.compositePipeline);
      compositePass.setBindGroup(0, this.compositeBindGroup);
      compositePass.draw(3);
      compositePass.end();
    }

    this.device.queue.submit([encoder.finish()]);
  }

  destroy() {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    try {
      this.densityTexture?.destroy();
      this.densityTexture = null;
      for (const buffer of [
        ...this.particleBuffers,
        this.scalarBuffer,
        this.vectorBuffer,
        this.simulationUniformBuffer,
        this.renderUniformBuffer,
      ]) {
        buffer.destroy();
      }
    } finally {
      try {
        this.context.unconfigure();
      } finally {
        this.device.destroy();
      }
    }
  }

  writeSimulationUniforms() {
    const data = new ArrayBuffer(SIMULATION_UNIFORM_BYTES);
    const view = new DataView(data);
    view.setFloat32(0, this.bounds.x, true);
    view.setFloat32(4, this.bounds.y, true);
    view.setFloat32(8, this.pointer?.x ?? 0, true);
    view.setFloat32(12, this.pointer?.y ?? 0, true);
    view.setUint32(16, this.particleCount, true);
    view.setUint32(20, this.pointer ? 1 : 0, true);
    view.setFloat32(24, DELTA_TIME, true);
    view.setFloat32(28, this.smoothingRadius, true);
    view.setFloat32(32, this.particleRadius, true);
    view.setFloat32(36, this.mass, true);
    view.setFloat32(40, REST_DENSITY, true);
    view.setFloat32(44, this.pressureMultiplier, true);
    view.setFloat32(48, VISCOSITY, true);
    view.setFloat32(52, GRAVITY, true);
    view.setFloat32(56, POINTER_RADIUS, true);
    view.setFloat32(60, this.pointerStrength, true);
    this.device.queue.writeBuffer(this.simulationUniformBuffer, 0, data);
  }

  writeRenderUniforms() {
    const data = new Float32Array(RENDER_UNIFORM_BYTES / Float32Array.BYTES_PER_ELEMENT);
    data.set([this.bounds.x, this.bounds.y, this.particleRadius, this.particleRadius * 5.0], 0);
    data.set(this.palette.ink, 4);
    data.set(this.palette.accent, 8);
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, data);
  }
}

const IDLE_DELAY_MS = 8_000;

export function restartSimulationForMotionPreference(simulation, reducedMotion) {
  simulation.reset({ settled: reducedMotion });
  simulation.setPointer(null);
  if (reducedMotion) {
    simulation.advance(32);
    simulation.render();
  }
}

export async function startSphHero(root) {
  const shell = root?.querySelector(".sph-fluid-shell");
  const canvas = shell?.querySelector("canvas");
  const fallback = shell?.querySelector(".sph-fluid-fallback");
  const restartButton = shell?.querySelector(".sph-fluid-restart");
  const controls = root?.querySelector(".sph-fluid-controls");
  const particleInput = controls?.querySelector("#sph-particles");
  const particleOutput = controls?.querySelector("#sph-particles-output");
  const pressureInput = controls?.querySelector("#sph-pressure");
  const pressureOutput = controls?.querySelector("#sph-pressure-output");
  const pointerForceInput = controls?.querySelector("#sph-pointer-force");
  const pointerForceOutput = controls?.querySelector("#sph-pointer-force-output");
  const renderModeInputs = controls
    ? [...controls.querySelectorAll('input[name="sph-render-mode"]')]
    : [];
  if (
    !shell || !canvas || !fallback || !restartButton || !controls ||
    !particleInput || !particleOutput || !pressureInput || !pressureOutput ||
    !pointerForceInput || !pointerForceOutput || renderModeInputs.length !== 2
  ) {
    return () => {};
  }

  const controlInputs = [
    particleInput,
    pressureInput,
    pointerForceInput,
    ...renderModeInputs,
  ];
  let profile = getSimulationProfile(window.innerWidth);
  const settings = {
    particleCount: profile.particleCount,
    pressureMultiplier: 0.55,
    pointerStrength: 5.0,
    renderMode: "balls",
  };
  let particleCountTouched = false;

  function setControlInputsDisabled(disabled) {
    for (const input of controlInputs) input.disabled = disabled;
  }

  function setOutput(output, value, fractionDigits) {
    const formatted = Number(value).toFixed(fractionDigits);
    output.value = formatted;
    output.textContent = formatted;
  }

  function readRange(input) {
    return Math.min(Number(input.max), Math.max(Number(input.min), Number(input.value)));
  }

  particleInput.value = String(settings.particleCount);
  particleOutput.value = String(settings.particleCount);
  particleOutput.textContent = String(settings.particleCount);
  setOutput(pressureOutput, settings.pressureMultiplier, 2);
  setOutput(pointerForceOutput, settings.pointerStrength, 1);
  for (const input of renderModeInputs) {
    input.checked = input.value === settings.renderMode;
  }
  setControlInputsDisabled(true);
  restartButton.disabled = true;

  const state = {
    inViewport: false,
    pageVisible: !document.hidden,
    reducedMotion: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    failed: false,
    idle: false,
  };
  let simulation = null;
  let animationFrame = null;
  let idleRemainingMs = IDLE_DELAY_MS;
  let previousFrameTime = null;
  let generation = 0;
  let disposed = false;
  let resizeObserver = null;
  let intersectionObserver = null;
  let pointerListenersAttached = false;
  let deviceErrorListener = null;
  let recreationRequested = false;
  let recreationPromise = null;
  let renderPending = false;

  function cancelAnimation() {
    if (animationFrame !== null) {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    previousFrameTime = null;
  }

  function disconnectObservers() {
    resizeObserver?.disconnect();
    intersectionObserver?.disconnect();
    resizeObserver = null;
    intersectionObserver = null;
  }

  function removeEventListeners() {
    document.removeEventListener("visibilitychange", onVisibilityChange);
    restartButton.removeEventListener("click", onRestart);
    particleInput.removeEventListener("input", onParticleInput);
    particleInput.removeEventListener("change", onParticleChange);
    pressureInput.removeEventListener("input", onPressureInput);
    pointerForceInput.removeEventListener("input", onPointerForceInput);
    for (const input of renderModeInputs) {
      input.removeEventListener("change", onRenderModeChange);
    }
    if (pointerListenersAttached) {
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerleave", clearPointer);
      canvas.removeEventListener("pointercancel", clearPointer);
      canvas.removeEventListener("pointerup", clearPointer);
      pointerListenersAttached = false;
    }
  }

  function destroyCurrentSimulation() {
    const current = simulation;
    const currentErrorListener = deviceErrorListener;
    simulation = null;
    deviceErrorListener = null;
    if (!current) {
      return;
    }
    if (currentErrorListener) {
      current.device.removeEventListener("uncapturederror", currentErrorListener);
    }
    current.destroy();
  }

  function fail(error) {
    if (state.failed || disposed) {
      return;
    }
    state.failed = true;
    generation += 1;
    cancelAnimation();
    disconnectObservers();
    removeEventListeners();
    try {
      destroyCurrentSimulation();
    } catch {
      // The failure UI still needs to be shown if cleanup itself fails.
    }
    canvas.hidden = true;
    restartButton.hidden = true;
    controls.hidden = true;
    fallback.hidden = false;
    console.warn("WebGPU fluid unavailable", error);
  }

  function reconcileAnimation() {
    if (
      renderPending &&
      !disposed &&
      !state.failed &&
      simulation &&
      state.inViewport &&
      state.pageVisible
    ) {
      try {
        simulation.render();
        renderPending = false;
      } catch (error) {
        fail(error);
        return;
      }
    }
    if (!disposed && simulation && shouldAnimate(state)) {
      if (animationFrame === null) {
        animationFrame = window.requestAnimationFrame(onAnimationFrame);
      }
    } else {
      cancelAnimation();
    }
  }

  function onAnimationFrame(frameTime) {
    animationFrame = null;
    if (!simulation || !shouldAnimate(state)) {
      reconcileAnimation();
      return;
    }
    try {
      simulation.advance(2);
      simulation.render();
    } catch (error) {
      fail(error);
      return;
    }
    const activeFrameTime = consumeActiveFrameTime(
      idleRemainingMs,
      previousFrameTime,
      frameTime,
    );
    idleRemainingMs = activeFrameTime.remainingMs;
    previousFrameTime = activeFrameTime.previousFrameTime;
    if (idleRemainingMs === 0) {
      state.idle = true;
      simulation.setPointer(null);
    }
    reconcileAnimation();
  }

  function onVisibilityChange() {
    state.pageVisible = !document.hidden;
    reconcileAnimation();
  }

  function onRestart() {
    if (!simulation || state.failed || disposed) {
      return;
    }
    restartButton.disabled = true;
    try {
      restartSimulationForMotionPreference(simulation, state.reducedMotion);
      previousFrameTime = null;
      if (state.reducedMotion) {
        state.idle = true;
      } else {
        idleRemainingMs = IDLE_DELAY_MS;
        state.idle = false;
      }
    } catch (error) {
      fail(error);
      return;
    }
    restartButton.disabled = false;
    reconcileAnimation();
  }

  function clearPointer() {
    simulation?.setPointer(null);
  }

  function onPointerMove(event) {
    if (!simulation) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      simulation.setPointer(null);
      return;
    }
    const point = clientPointToSimulation(event.clientX, event.clientY, rect);
    const bounds = { x: canvas.width / canvas.height, y: 1 };
    const inBounds = point.x >= 0 && point.x <= bounds.x && point.y >= 0 && point.y <= bounds.y;
    if (!inBounds) {
      simulation.setPointer(null);
      return;
    }
    simulation.setPointer(point);
    idleRemainingMs = IDLE_DELAY_MS;
    previousFrameTime = null;
    state.idle = false;
    reconcileAnimation();
  }

  function onParticleInput() {
    const value = Math.round(readRange(particleInput));
    particleOutput.value = String(value);
    particleOutput.textContent = String(value);
  }

  function onParticleChange() {
    const particleCount = Math.round(readRange(particleInput));
    particleCountTouched = true;
    particleOutput.value = String(particleCount);
    particleOutput.textContent = String(particleCount);
    if (particleCount === settings.particleCount) return;
    settings.particleCount = particleCount;
    profile = { ...profile, particleCount };
    void recreateSimulation();
  }

  function onPressureInput() {
    const pressureMultiplier = readRange(pressureInput);
    settings.pressureMultiplier = pressureMultiplier;
    setOutput(pressureOutput, pressureMultiplier, 2);
    if (!simulation) return;
    try {
      simulation.setParameters({ pressureMultiplier });
      if (!state.reducedMotion) {
        idleRemainingMs = IDLE_DELAY_MS;
        previousFrameTime = null;
        state.idle = false;
        reconcileAnimation();
      }
    } catch (error) {
      fail(error);
    }
  }

  function onPointerForceInput() {
    const pointerStrength = readRange(pointerForceInput);
    settings.pointerStrength = pointerStrength;
    setOutput(pointerForceOutput, pointerStrength, 1);
    if (!simulation) return;
    try {
      simulation.setParameters({ pointerStrength });
    } catch (error) {
      fail(error);
    }
  }

  function onRenderModeChange(event) {
    const input = event.currentTarget;
    if (!input.checked) return;
    const renderMode = input.value;
    settings.renderMode = renderMode;
    if (!simulation) return;
    try {
      simulation.setRenderMode(renderMode);
      renderPending = true;
      reconcileAnimation();
    } catch (error) {
      fail(error);
    }
  }

  function resizeCanvas(cssWidth, resizeSimulation = true) {
    const metrics = getCanvasMetrics(cssWidth, profile.cssHeight, window.devicePixelRatio);
    if (canvas.width !== metrics.width) {
      canvas.width = metrics.width;
    }
    if (canvas.height !== metrics.height) {
      canvas.height = metrics.height;
    }
    if (simulation && resizeSimulation) {
      simulation.resize(metrics.width, metrics.height);
      renderPending = true;
    }
  }

  async function runRecreationLoop() {
    while (recreationRequested && !disposed && !state.failed) {
      recreationRequested = false;
      await createRequestedSimulation();
    }
  }

  async function createRequestedSimulation() {
    const requestedProfile = profile;
    const requestedGeneration = generation + 1;
    generation = requestedGeneration;
    cancelAnimation();
    restartButton.disabled = true;
    setControlInputsDisabled(true);
    try {
      destroyCurrentSimulation();
    } catch (error) {
      fail(error);
      return;
    }

    let created;
    try {
      created = await SphSimulation.create(canvas, requestedProfile, {
        settled: state.reducedMotion,
        pressureMultiplier: settings.pressureMultiplier,
        pointerStrength: settings.pointerStrength,
        renderMode: settings.renderMode,
      });
    } catch (error) {
      if (!disposed && !state.failed && requestedGeneration === generation) {
        if (requestedProfile.particleCount !== profile.particleCount) {
          recreationRequested = true;
        } else {
          fail(error);
        }
      }
      return;
    }

    if (
      disposed ||
      state.failed ||
      requestedGeneration !== generation ||
      requestedProfile.particleCount !== profile.particleCount
    ) {
      try {
        created.destroy();
      } catch {
        // A stale simulation must not affect the active generation.
      }
      if (!disposed && !state.failed && requestedGeneration === generation) {
        recreationRequested = true;
      }
      return;
    }

    recreationRequested = false;
    simulation = created;
    try {
      deviceErrorListener = (event) => {
        if (simulation === created) {
          fail(event.error ?? event);
        }
      };
      created.device.addEventListener("uncapturederror", deviceErrorListener);
      created.device.lost.then((info) => {
        if (simulation === created) {
          fail(info);
        }
      });
      created.resize(canvas.width, canvas.height);
      if (state.reducedMotion) {
        created.advance(32);
        created.render();
        state.idle = true;
      } else {
        idleRemainingMs = IDLE_DELAY_MS;
        previousFrameTime = null;
        state.idle = false;
      }
    } catch (error) {
      fail(error);
      return;
    }
    restartButton.disabled = false;
    setControlInputsDisabled(false);
    reconcileAnimation();
  }

  function recreateSimulation() {
    recreationRequested = true;
    if (!recreationPromise) {
      recreationPromise = runRecreationLoop().finally(() => {
        recreationPromise = null;
        if (recreationRequested && !disposed && !state.failed) {
          void recreateSimulation();
        }
      });
    }
    return recreationPromise;
  }

  function onResize(entries) {
    const entry = entries.find((candidate) => candidate.target === root) ?? entries[0];
    const cssWidth = entry?.contentRect.width ?? root.getBoundingClientRect().width;
    const responsiveProfile = getSimulationProfile(window.innerWidth);
    const particleCount = particleCountTouched
      ? settings.particleCount
      : responsiveProfile.particleCount;
    const nextProfile = { ...responsiveProfile, particleCount };
    const populationChanged = nextProfile.particleCount !== profile.particleCount;
    if (!particleCountTouched && settings.particleCount !== particleCount) {
      settings.particleCount = particleCount;
      particleInput.value = String(particleCount);
      particleOutput.value = String(particleCount);
      particleOutput.textContent = String(particleCount);
    }
    profile = nextProfile;
    try {
      resizeCanvas(cssWidth, !populationChanged);
      if (populationChanged) {
        void recreateSimulation();
      } else {
        reconcileAnimation();
      }
    } catch (error) {
      fail(error);
    }
  }

  function teardown() {
    if (disposed) {
      return;
    }
    restartButton.disabled = true;
    setControlInputsDisabled(true);
    disposed = true;
    generation += 1;
    cancelAnimation();
    disconnectObservers();
    removeEventListeners();
    try {
      destroyCurrentSimulation();
    } catch {
      // Teardown is best-effort and does not switch to the failure UI.
    }
  }

  try {
    resizeCanvas(root.getBoundingClientRect().width, false);
    resizeObserver = new ResizeObserver(onResize);
    resizeObserver.observe(root);
    intersectionObserver = new IntersectionObserver((entries) => {
      const entry = entries.find((candidate) => candidate.target === shell) ?? entries[0];
      state.inViewport = entry?.isIntersecting ?? false;
      reconcileAnimation();
    });
    intersectionObserver.observe(shell);
    document.addEventListener("visibilitychange", onVisibilityChange);
    restartButton.addEventListener("click", onRestart);
    particleInput.addEventListener("input", onParticleInput);
    particleInput.addEventListener("change", onParticleChange);
    pressureInput.addEventListener("input", onPressureInput);
    pointerForceInput.addEventListener("input", onPointerForceInput);
    for (const input of renderModeInputs) {
      input.addEventListener("change", onRenderModeChange);
    }

    if (!state.reducedMotion) {
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("pointerleave", clearPointer);
      canvas.addEventListener("pointercancel", clearPointer);
      canvas.addEventListener("pointerup", clearPointer);
      pointerListenersAttached = true;
    }

    await recreateSimulation();
  } catch (error) {
    fail(error);
  }

  return teardown;
}

if (typeof document !== "undefined") {
  const root = document.querySelector("[data-sph-fluid]");
  if (root) startSphHero(root);
}
