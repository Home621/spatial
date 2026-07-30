# Rangefinder — spatial photos in the browser

Upload a flat photo, get an iOS-Spatial-Photo-style image back: depth is
estimated **locally on-device** (Depth Anything V2, running client-side via
`transformers.js`/ONNX Runtime Web — no server, nothing uploaded), and the
photo is rendered as a depth-displaced 3D mesh that reacts to your mouse
(desktop) or gyroscope (phone).

## Run it

Because the app uses native ES module imports, it needs to be served over
`http://`, not opened directly as a `file://` (browsers block module imports
from the filesystem). From this folder:

```bash
python3 -m http.server 8000
# or: npx serve .
```

Then open `http://localhost:8000` — on a phone, use your computer's LAN IP
(e.g. `http://192.168.1.23:8000`) so the phone can reach it, or `ngrok http 8000`.

**On first use** it downloads the depth model (~27 MB, quantized) straight
from Hugging Face's CDN and caches it in the browser — so you need internet
for that one time, and every device tries again once (per browser/origin).
After that, it's fully offline and instant.

## How it works

- `main.js` reads your photo into a canvas, feeds it to a `depth-estimation`
  pipeline (`onnx-community/depth-anything-v2-small`, 8-bit quantized) which
  runs via WebAssembly (or WebGPU if your browser supports it).
- The resulting grayscale depth map becomes a texture. A subdivided plane
  mesh is displaced along Z per-vertex based on that texture in a small
  GLSL vertex shader — near things are pushed toward the camera.
- A virtual camera gently orbits around the mesh, driven by mouse position,
  `deviceorientation` events (gyroscope), or a slow idle drift when you're
  not touching it — producing real perspective parallax (Three.js/WebGL),
  the same trick behind Apple/Facebook "3D photo" viewers.

## Things worth knowing

- **iOS Safari** requires a tap to grant motion-sensor access (Apple's
  privacy rule) — that's what the "Enable motion" button is for. It only
  appears on touch devices.
- **Edge stretching**: at extreme tilt angles you'll see foreground edges
  stretch into the background — that's inherent to displacing a single flat
  mesh with no hidden-surface data, and it's the same artifact you'll see in
  every tool that does this trick without multi-layer inpainting. Keep the
  depth slider modest, or extend the shader to fade opacity near depth
  discontinuities if you want to hide it further.
- **If the effect looks inverted** (background bulges toward you instead of
  the subject), hit "Invert" — different photos occasionally confuse the
  depth model's near/far convention.
- To trade quality for size/speed, swap `MODEL_ID` in `main.js` for
  `onnx-community/depth-anything-v2-base` (better quality, ~100 MB in q8)
  or keep `-small` (fast, ~27 MB).

## Files

- `index.html` — page shell, import map (loads Three.js + transformers.js
  from jsDelivr at runtime)
- `style.css` — visual design
- `main.js` — upload handling, depth pipeline, Three.js scene, input handling
