import * as THREE from 'three';
import { pipeline, RawImage, env } from '@huggingface/transformers';
import { Client } from '@gradio/client';

// GitHub Pages (and most static hosts) can't set the COOP/COEP headers that
// onnxruntime-web's multi-threaded WASM path wants. Depending on the exact
// build, that can throw instead of gracefully degrading — so force
// single-threaded WASM up front. Slightly slower, much more compatible.
env.backends.onnx.wasm.numThreads = 1;

/* ----------------------------------------------------------------------
   Rangefinder — turns a flat photo into a mouse/gyro-reactive "spatial
   photo" using a locally-run monocular depth model (Depth Anything V2,
   small, running fully client-side via transformers.js/ONNX Runtime Web)
   plus a depth-displaced mesh rendered with Three.js.

   Nothing here ever leaves the browser: the photo is never uploaded,
   the depth model runs on-device, and the only network traffic is the
   one-time model download (cached by the browser afterwards).
   ---------------------------------------------------------------------- */

const MODEL_ID = 'onnx-community/depth-anything-v2-small';
const MAX_IMAGE_DIM = 1600;      // cap texture size for perf on mobile GPUs
const MAX_ANGLE = 0.22;          // radians of camera orbit at full tilt/mouse throw
const IDLE_TIMEOUT = 2200;       // ms of no input before ambient drift kicks in

// Set this to your own free Hugging Face Space (see /space in this project)
// to have depth computed server-side first, e.g. 'yourname/depth-anything-v2-small'.
// Leave it '' to skip straight to on-device inference.
const REMOTE_SPACE = '';
const REMOTE_TIMEOUT_MS = 20000; // free Spaces sleep when idle; give a cold one time to wake

// ---- DOM ----
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const canvasFrame = document.getElementById('canvasFrame');
const glCanvas = document.getElementById('glCanvas');
const depthPreview = document.getElementById('depthPreview');
const hudStatus = document.getElementById('hudStatus');
const statusOverlay = document.getElementById('statusOverlay');
const statusText = document.getElementById('statusText');
const spinnerEl = document.getElementById('spinner');
const progressTrack = document.getElementById('progressTrack');
const progressFill = document.getElementById('progressFill');
const retryBtn = document.getElementById('retryBtn');
const controlRail = document.getElementById('controlRail');
const newPhotoBtn = document.getElementById('newPhotoBtn');
const depthSlider = document.getElementById('depthSlider');
const invertBtn = document.getElementById('invertBtn');
const depthMapBtn = document.getElementById('depthMapBtn');
const motionBtn = document.getElementById('motionBtn');

// ---- state ----
let estimator = null;
let depthCanvasEl = null;
let renderer, scene, camera, mesh, material, planeW, planeH, baseDistance;
let targetX = 0, targetY = 0;     // -1..1, from pointer or gyro
let currentX = 0, currentY = 0;   // smoothed
let lastInputAt = 0;
let baseBeta = null, baseGamma = null;
let reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ============================== UI: upload ==============================

['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('dragover'); })
);
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
});
newPhotoBtn.addEventListener('click', () => fileInput.click());

async function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  try {
    const img = await loadImage(file);
    const canvas = drawToCanvas(img);
    showViewer();
    setStatus('Reading depth from the photo…', true);
    setProgress(null); // indeterminate until we know sizes

    const depth = await estimateDepth(canvas);
    depthCanvasEl = depth;
    depthPreview.src = depth.toDataURL('image/png');

    buildScene(canvas, depth);
    hideStatus();
    controlRail.hidden = false;
    hudStatus.textContent = depthSource === 'remote'
      ? `${canvas.width}×${canvas.height} · depth via your Space (remote)`
      : `${canvas.width}×${canvas.height} · depth-anything-v2-small (q8) · on-device (${estimatorDevice})`;
    maybeShowMotionButton();
  } catch (err) {
    console.error(err);
    const hint = err?.message ? ` (${err.message})` : '';
    showError(`Couldn't generate a depth map${hint}. Try a different photo, or check the console for details.`);
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function drawToCanvas(img) {
  let { naturalWidth: w, naturalHeight: h } = img;
  if (Math.max(w, h) > MAX_IMAGE_DIM) {
    const scale = MAX_IMAGE_DIM / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(img, 0, 0, w, h);
  return canvas;
}

function showViewer() {
  dropzone.hidden = true;
  canvasFrame.hidden = false;
}

// ============================== depth model ==============================

let estimatorDevice = 'wasm';
const progressByFile = new Map();

async function getEstimator() {
  if (estimator) return estimator;

  estimatorDevice = (typeof navigator !== 'undefined' && navigator.gpu) ? 'webgpu' : 'wasm';
  setStatus('Downloading depth model (first run only)…', true);

  const onProgress = (p) => {
    if (p.status === 'progress' && p.file) {
      progressByFile.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
      let loaded = 0, total = 0;
      for (const v of progressByFile.values()) { loaded += v.loaded; total += v.total; }
      if (total > 0) setProgress(Math.min(100, (loaded / total) * 100));
    }
  };

  try {
    estimator = await pipeline('depth-estimation', MODEL_ID, {
      device: estimatorDevice,
      dtype: 'q8',
      progress_callback: onProgress,
    });
  } catch (err) {
    // Fall back to WASM if WebGPU init failed on this browser.
    console.warn('Falling back to wasm backend:', err);
    estimatorDevice = 'wasm';
    estimator = await pipeline('depth-estimation', MODEL_ID, {
      device: 'wasm',
      dtype: 'q8',
      progress_callback: onProgress,
    });
  }
  return estimator;
}

let depthSource = 'local'; // 'remote' | 'local' — which path produced the last depth map

async function estimateDepth(colorCanvas) {
  if (REMOTE_SPACE) {
    setStatus('Asking your Space to compute depth… (can take ~30s if it was asleep)', true);
    setProgress(null);
    const remote = await tryRemoteDepth(colorCanvas);
    if (remote) {
      depthSource = 'remote';
      return remote;
    }
    setStatus('Space unavailable — running the depth model on this device instead…', true);
  }
  depthSource = 'local';
  return estimateDepthLocal(colorCanvas);
}

async function estimateDepthLocal(colorCanvas) {
  const est = await getEstimator();
  setStatus('Running depth estimation on this device…', true);
  setProgress(null);
  const rawImage = RawImage.fromCanvas(colorCanvas);
  const { depth } = await est(rawImage);
  return depth.toCanvas();
}

// Calls a Gradio Space (see the /space folder) that runs the same depth
// model server-side. Returns a canvas on success, or null on any failure —
// caller falls back to local inference either way, so this never throws.
async function tryRemoteDepth(colorCanvas) {
  try {
    const blob = await new Promise(resolve => colorCanvas.toBlob(resolve, 'image/jpeg', 0.92));
    const file = new File([blob], 'photo.jpg', { type: 'image/jpeg' });

    const client = await withTimeout(Client.connect(REMOTE_SPACE), REMOTE_TIMEOUT_MS);
    const result = await withTimeout(
      client.predict('/predict', { image: file }),
      REMOTE_TIMEOUT_MS
    );

    const url = result?.data?.[0]?.url;
    if (!url) throw new Error('Space returned no depth image');

    const img = await loadImageFromURL(url);
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    return canvas;
  } catch (err) {
    console.warn('Remote depth via Space failed, falling back to local:', err);
    return null;
  }
}

function loadImageFromURL(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function withTimeout(promise, ms) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timed out')), ms);
    promise.then(
      v => { clearTimeout(timer); resolve(v); },
      err => { clearTimeout(timer); reject(err); }
    );
  });
}

// ============================== Three.js scene ==============================

function buildScene(colorCanvas, depthCanvas) {
  const aspect = colorCanvas.width / colorCanvas.height;
  planeH = 3;
  planeW = 3 * aspect;

  if (!renderer) {
    renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    window.addEventListener('resize', onResize);
    new ResizeObserver(onResize).observe(canvasFrame);
  }

  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    material.uniforms.uColor.value.dispose();
    material.uniforms.uDepth.value.dispose();
    material.dispose();
  }

  const colorTex = new THREE.CanvasTexture(colorCanvas);
  colorTex.colorSpace = THREE.SRGBColorSpace;
  const depthTex = new THREE.CanvasTexture(depthCanvas);

  const geo = new THREE.PlaneGeometry(planeW, planeH, 220, 220);

  material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: colorTex },
      uDepth: { value: depthTex },
      uDepthScale: { value: sliderToScale(depthSlider.value) },
      uInvert: { value: 0 },
    },
    vertexShader: `
      varying vec2 vUv;
      uniform sampler2D uDepth;
      uniform float uDepthScale;
      uniform float uInvert;
      void main() {
        vUv = uv;
        float d = texture2D(uDepth, uv).r;
        if (uInvert > 0.5) d = 1.0 - d;
        vec3 pos = position;
        pos.z += d * uDepthScale;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `,
    fragmentShader: `
      varying vec2 vUv;
      uniform sampler2D uColor;
      void main() {
        gl_FragColor = texture2D(uColor, vUv);
      }
    `,
  });

  mesh = new THREE.Mesh(geo, material);
  scene.add(mesh);

  onResize();
  if (!animating) { animating = true; requestAnimationFrame(animate); }
}

function sliderToScale(v) {
  // slider is 0..120 -> plane-space displacement
  return (Number(v) / 120) * 1.0;
}

function onResize() {
  const w = canvasFrame.clientWidth || 1;
  const h = canvasFrame.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  fitCamera(w / h);
}

function fitCamera(canvasAspect) {
  const fovRad = (camera.fov * Math.PI) / 180;
  const distV = (planeH / 2) / Math.tan(fovRad / 2);
  const distH = (planeW / 2) / Math.tan(fovRad / 2) / canvasAspect;
  baseDistance = Math.max(distV, distH) * 1.06;
}

// ============================== input: mouse / touch / gyro ==============================

canvasFrame.addEventListener('pointermove', e => {
  const rect = canvasFrame.getBoundingClientRect();
  targetX = (((e.clientX - rect.left) / rect.width) * 2 - 1);
  targetY = (((e.clientY - rect.top) / rect.height) * 2 - 1);
  lastInputAt = performance.now();
});
canvasFrame.addEventListener('pointerleave', () => { targetX = 0; targetY = 0; });

function handleOrientation(e) {
  if (e.beta === null || e.gamma === null) return;
  if (baseBeta === null) { baseBeta = e.beta; baseGamma = e.gamma; }
  const dGamma = clamp((e.gamma - baseGamma) / 26, -1, 1);
  const dBeta = clamp((e.beta - baseBeta) / 26, -1, 1);
  targetX = dGamma;
  targetY = dBeta;
  lastInputAt = performance.now();
}

function maybeShowMotionButton() {
  const isTouch = matchMedia('(pointer: coarse)').matches;
  if (!isTouch || typeof DeviceOrientationEvent === 'undefined') { motionBtn.hidden = true; return; }
  motionBtn.hidden = false;
}

motionBtn.addEventListener('click', async () => {
  try {
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
      const res = await DeviceOrientationEvent.requestPermission();
      if (res !== 'granted') return;
    }
    window.addEventListener('deviceorientation', handleOrientation);
    motionBtn.hidden = true;
  } catch (err) {
    console.warn('Motion permission error', err);
  }
});

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ============================== controls ==============================

depthSlider.addEventListener('input', () => {
  if (material) material.uniforms.uDepthScale.value = sliderToScale(depthSlider.value);
});

invertBtn.addEventListener('click', () => {
  const pressed = invertBtn.getAttribute('aria-pressed') === 'true';
  invertBtn.setAttribute('aria-pressed', String(!pressed));
  if (material) material.uniforms.uInvert.value = pressed ? 0 : 1;
});

depthMapBtn.addEventListener('click', () => {
  const pressed = depthMapBtn.getAttribute('aria-pressed') === 'true';
  depthMapBtn.setAttribute('aria-pressed', String(!pressed));
  depthPreview.hidden = pressed;
  glCanvas.style.visibility = pressed ? 'visible' : 'hidden';
});

// ============================== render loop ==============================

let animating = false;

function animate(t) {
  requestAnimationFrame(animate);
  if (!renderer || !camera) return;

  const idle = performance.now() - lastInputAt > IDLE_TIMEOUT;
  let tx = targetX, ty = targetY;
  if (idle && !reduceMotion) {
    tx = Math.sin(t * 0.00028) * 0.35;
    ty = Math.cos(t * 0.00021) * 0.22;
  } else if (idle) {
    tx = 0; ty = 0;
  }

  currentX += (tx - currentX) * 0.06;
  currentY += (ty - currentY) * 0.06;

  const ax = currentX * MAX_ANGLE;
  const ay = currentY * MAX_ANGLE;
  camera.position.x = Math.sin(ax) * baseDistance;
  camera.position.y = Math.sin(-ay) * baseDistance;
  camera.position.z = Math.cos(ax) * Math.cos(ay) * baseDistance;
  camera.lookAt(0, 0, 0);

  renderer.render(scene, camera);
}

// ============================== status helpers ==============================

function setStatus(msg, show) {
  statusText.textContent = msg;
  statusOverlay.hidden = !show;
}

function showError(msg) {
  statusText.textContent = msg;
  spinnerEl.hidden = true;
  progressTrack.hidden = true;
  retryBtn.hidden = false;
  statusOverlay.hidden = false;
}

function hideStatus() {
  statusOverlay.hidden = true;
  spinnerEl.hidden = false;
  progressTrack.hidden = false;
  retryBtn.hidden = true;
}

function setProgress(pct) {
  progressFill.style.width = pct === null ? '6%' : `${pct}%`;
}

retryBtn.addEventListener('click', () => {
  hideStatus();
  canvasFrame.hidden = true;
  controlRail.hidden = true;
  dropzone.hidden = false;
  fileInput.value = '';
});
