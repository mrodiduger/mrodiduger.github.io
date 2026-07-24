import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  COMPOSITE_SHADER,
  COMPUTE_SHADER,
  DENSITY_RENDER_SHADER,
  RENDER_SHADER,
  SIMULATION_UNIFORM_BYTES,
  SphSimulation,
  clientPointToSimulation,
  consumeActiveFrameTime,
  createInitialParticles,
  getCanvasMetrics,
  getSimulationProfile,
  restartSimulationForMotionPreference,
  shouldAnimate,
} from "../assets/js/sph-fluid.mjs";

test("places a restartable fluid strip and four controls below the header", async () => {
  const index = await readFile(new URL("../index.qmd", import.meta.url), "utf8");
  const css = await readFile(new URL("../assets/css/site.css", import.meta.url), "utf8");
  const script = await readFile(new URL("../assets/js/sph-fluid.mjs", import.meta.url), "utf8");
  const blockPosition = index.indexOf('<div class="sph-fluid-block" data-sph-fluid>');
  const shellPosition = index.indexOf('<div class="sph-fluid-shell">');
  const controlsPosition = index.indexOf('<div class="sph-fluid-controls"');
  const heroPosition = index.indexOf('<section class="hero"');

  assert.ok(
    blockPosition > -1 && blockPosition < shellPosition &&
      shellPosition < controlsPosition && controlsPosition < heroPosition,
    "fluid shell and controls must precede the hero in that order",
  );
  assert.match(
    index,
    /<button\s+class="sph-fluid-restart"\s+type="button"\s+aria-label="Restart fluid simulation"\s+disabled\s*>\s*restart\s*<\/button>/,
  );
  assert.match(index, /class="sph-fluid-controls"\s+role="group"\s+aria-label="Fluid simulation parameters"/);
  assert.match(index, /id="sph-particles"\s+type="range"\s+min="200"\s+max="2400"\s+step="8"\s+value="600"\s+disabled/);
  assert.match(index, /id="sph-pressure"\s+type="range"\s+min="0\.1"\s+max="1\.2"\s+step="0\.05"\s+value="0\.55"\s+disabled/);
  assert.match(index, /id="sph-pointer-force"\s+type="range"\s+min="0"\s+max="20"\s+step="0\.5"\s+value="5"\s+disabled/);
  assert.equal((index.match(/class="sph-fluid-control"/g) ?? []).length, 3);
  assert.equal((index.match(/name="sph-render-mode"/g) ?? []).length, 2);
  assert.match(index, /name="sph-render-mode"\s+value="balls"\s+checked\s+disabled/);
  assert.match(index, /name="sph-render-mode"\s+value="fluid"\s+disabled/);
  assert.match(css, /\.sph-fluid-controls\s*\{[^}]*grid-template-columns:\s*repeat\(4, minmax\(0, 1fr\)\);/s);
  assert.match(css, /\.sph-fluid-mode input:checked \+ span\s*\{[^}]*color:\s*var\(--ink\);/s);
  assert.match(css, /\.sph-fluid-mode input:focus-visible \+ span\s*\{[^}]*outline:\s*1px solid var\(--ink\);/s);
  assert.match(css, /\.sph-fluid-controls\[hidden\]\s*\{\s*display:\s*none;/);
  assert.match(css, /@media \(max-width: 720px\)[\s\S]*?\.sph-fluid-controls\s*\{[^}]*grid-template-columns:\s*1fr;/);
  assert.match(script, /renderMode:\s*"balls"/);
  assert.match(script, /simulation\.setRenderMode\(renderMode\)/);
  assert.match(script, /let renderPending = false;/);
  assert.doesNotMatch(script, /resizeRenderPending/);
});

test("hides the failed canvas so the fallback remains visible", async () => {
  const css = await readFile(new URL("../assets/css/site.css", import.meta.url), "utf8");
  assert.ok(
    /#sph-fluid-canvas\[hidden\]\s*\{\s*display:\s*none;\s*\}/.test(css),
    "site.css must explicitly hide #sph-fluid-canvas[hidden]",
  );
});

test("animates only while visible, active, and motion is allowed", () => {
  const active = {
    inViewport: true,
    pageVisible: true,
    reducedMotion: false,
    failed: false,
    idle: false,
  };
  assert.equal(shouldAnimate(active), true);
  assert.equal(shouldAnimate({ ...active, inViewport: false }), false);
  assert.equal(shouldAnimate({ ...active, pageVisible: false }), false);
  assert.equal(shouldAnimate({ ...active, reducedMotion: true }), false);
  assert.equal(shouldAnimate({ ...active, failed: true }), false);
  assert.equal(shouldAnimate({ ...active, idle: true }), false);
});

test("counts only consecutive active-frame time toward idle", () => {
  const resumed = consumeActiveFrameTime(8_000, null, 60_000);
  assert.deepEqual(resumed, { remainingMs: 8_000, previousFrameTime: 60_000 });

  const active = consumeActiveFrameTime(
    resumed.remainingMs,
    resumed.previousFrameTime,
    65_000,
  );
  assert.deepEqual(active, { remainingMs: 3_000, previousFrameTime: 65_000 });

  const expired = consumeActiveFrameTime(
    active.remainingMs,
    active.previousFrameTime,
    68_000,
  );
  assert.deepEqual(expired, { remainingMs: 0, previousFrameTime: 68_000 });
});

test("selects exact wide and narrow budgets at 720px", () => {
  assert.deepEqual(getSimulationProfile(721), { particleCount: 600, cssHeight: 240 });
  assert.deepEqual(getSimulationProfile(720), { particleCount: 384, cssHeight: 168 });
});

test("caps device pixel ratio at two", () => {
  assert.deepEqual(getCanvasMetrics(320, 168, 3), {
    width: 640,
    height: 336,
    pixelRatio: 2,
  });
});

test("maps client coordinates into an aspect-correct simulation domain", () => {
  const point = clientPointToSimulation(250, 100, {
    left: 50,
    top: 25,
    width: 400,
    height: 150,
  });
  assert.deepEqual(point, { x: 4 / 3, y: 0.5 });
});

test("creates finite particle state inside the requested bounds", () => {
  const particles = createInitialParticles(384, { x: 2, y: 1 }, false, () => 0.5);
  assert.equal(particles.length, 384 * 4);
  for (let index = 0; index < particles.length; index += 4) {
    assert.ok(particles[index] > 0 && particles[index] < 2);
    assert.ok(particles[index + 1] > 0 && particles[index + 1] < 1);
    assert.equal(particles[index + 2], 0);
    assert.equal(particles[index + 3], 0);
  }
});

test("resets both existing particle buffers without allocating GPU resources", () => {
  const particleBuffers = [{ label: "a" }, { label: "b" }];
  const writes = [];
  const device = {
    queue: {
      writeBuffer(buffer, offset, data) {
        writes.push({ buffer, offset, data });
      },
    },
    createBuffer() {
      assert.fail("reset must not allocate a buffer");
    },
  };
  const simulation = new SphSimulation({
    device,
    particleCount: 4,
    particleBuffers,
    bounds: { x: 2, y: 1 },
  });
  let simulationUniformWrites = 0;
  let renderUniformWrites = 0;
  simulation.writeSimulationUniforms = () => { simulationUniformWrites += 1; };
  simulation.writeRenderUniforms = () => { renderUniformWrites += 1; };
  simulation.currentParticleIndex = 1;
  simulation.pointer = { x: 1, y: 0.5 };

  simulation.reset({ settled: false });

  assert.deepEqual(writes.map(({ buffer }) => buffer), particleBuffers);
  assert.deepEqual(writes.map(({ data }) => data.length), [16, 16]);
  assert.equal(simulation.currentParticleIndex, 0);
  assert.equal(simulation.pointer, null);
  assert.equal(simulationUniformWrites, 1);
  assert.equal(renderUniformWrites, 1);
});

test("updates pressure and pointer force without allocating GPU resources", () => {
  const device = {
    createBuffer() {
      assert.fail("parameter updates must not allocate a buffer");
    },
  };
  const simulation = new SphSimulation({ device });
  let uniformWrites = 0;
  simulation.writeSimulationUniforms = () => { uniformWrites += 1; };

  simulation.setParameters({ pressureMultiplier: 0.8, pointerStrength: 7.5 });

  assert.equal(simulation.pressureMultiplier, 0.8);
  assert.equal(simulation.pointerStrength, 7.5);
  assert.equal(uniformWrites, 1);
});

test("switches render mode without allocating GPU resources", () => {
  const device = {
    createBindGroup() {
      assert.fail("mode changes must not allocate a bind group");
    },
    createTexture() {
      assert.fail("mode changes must not allocate a texture");
    },
  };
  const simulation = new SphSimulation({ device });

  assert.equal(simulation.renderMode, "balls");
  simulation.setRenderMode("fluid");
  assert.equal(simulation.renderMode, "fluid");
});

test("restarts normal and reduced-motion simulations without recreation", () => {
  const calls = [];
  const simulation = {
    reset(options) { calls.push(["reset", options]); },
    setPointer(value) { calls.push(["pointer", value]); },
    advance(steps) { calls.push(["advance", steps]); },
    render() { calls.push(["render"]); },
  };

  restartSimulationForMotionPreference(simulation, false);
  assert.deepEqual(calls, [
    ["reset", { settled: false }],
    ["pointer", null],
  ]);

  calls.length = 0;
  restartSimulationForMotionPreference(simulation, true);
  assert.deepEqual(calls, [
    ["reset", { settled: true }],
    ["pointer", null],
    ["advance", 32],
    ["render"],
  ]);
});

test("declares the three SPH compute entry points", () => {
  assert.match(COMPUTE_SHADER, /fn densityPressure\(/);
  assert.match(COMPUTE_SHADER, /fn forces\(/);
  assert.match(COMPUTE_SHADER, /fn integrate\(/);
  assert.match(COMPUTE_SHADER, /for \(var other = 0u; other < uniforms\.particleCount/);
});

test("declares balls, density, and composite render shaders", async () => {
  const source = await readFile(new URL("../assets/js/sph-fluid.mjs", import.meta.url), "utf8");
  assert.match(RENDER_SHADER, /@builtin\(instance_index\)/);
  assert.match(RENDER_SHADER, /discard/);
  assert.match(DENSITY_RENDER_SHADER, /uniforms\.surfaceRadius/);
  assert.match(DENSITY_RENDER_SHADER, /pow\(max\(1\.0 - distance, 0\.0\), 2\.0\)/);
  assert.match(COMPOSITE_SHADER, /textureSample\(densityTexture, densitySampler, input\.uv\)\.r/);
  assert.match(COMPOSITE_SHADER, /smoothstep\(0\.42, 0\.62, density\)/);
  assert.match(source, /const DENSITY_TEXTURE_FORMAT = "rgba8unorm";/);
  assert.match(source, /color:\s*\{ srcFactor: "one", dstFactor: "one" \}/);
  assert.equal(SIMULATION_UNIFORM_BYTES, 64);
  assert.equal(typeof SphSimulation.create, "function");
});

test("releases each owned GPU device exactly once", async (t) => {
  const bufferDestroyCalls = Array(6).fill(0);
  const buffers = bufferDestroyCalls.map((_, index) => ({
    destroy() {
      bufferDestroyCalls[index] += 1;
    },
  }));
  const context = {
    unconfigureCalls: 0,
    unconfigure() {
      this.unconfigureCalls += 1;
    },
  };
  const device = {
    destroyCalls: 0,
    destroy() {
      this.destroyCalls += 1;
    },
  };
  const densityTexture = {
    destroyCalls: 0,
    destroy() {
      this.destroyCalls += 1;
    },
  };
  const simulation = new SphSimulation({
    context,
    device,
    particleBuffers: buffers.slice(0, 2),
    scalarBuffer: buffers[2],
    vectorBuffer: buffers[3],
    simulationUniformBuffer: buffers[4],
    renderUniformBuffer: buffers[5],
    densityTexture,
  });

  simulation.destroy();
  simulation.destroy();

  const failedDevice = {
    destroyCalls: 0,
    destroy() {
      this.destroyCalls += 1;
    },
  };
  const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  t.after(() => {
    if (navigatorDescriptor) {
      Object.defineProperty(globalThis, "navigator", navigatorDescriptor);
    } else {
      delete globalThis.navigator;
    }
  });
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          return {
            async requestDevice() {
              return failedDevice;
            },
          };
        },
      },
    },
  });

  await assert.rejects(
    SphSimulation.create({ getContext: () => null }, { particleCount: 1 }),
    /No WebGPU canvas context is available/,
  );
  assert.deepEqual(
    {
      bufferDestroyCalls,
      densityTextureDestroyCalls: densityTexture.destroyCalls,
      contextUnconfigureCalls: context.unconfigureCalls,
      deviceDestroyCalls: device.destroyCalls,
      failedDeviceDestroyCalls: failedDevice.destroyCalls,
    },
    {
      bufferDestroyCalls: Array(6).fill(1),
      densityTextureDestroyCalls: 1,
      contextUnconfigureCalls: 1,
      deviceDestroyCalls: 1,
      failedDeviceDestroyCalls: 1,
    },
  );
});
