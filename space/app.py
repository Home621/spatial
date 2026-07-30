"""
Depth estimation API — deploy this as a free Hugging Face Space (CPU basic,
no GPU needed, no payment info required) and the spatial-photo web app will
call it before falling back to running the model on-device.

Deploy:
  1. huggingface.co/new-space -> pick "Gradio" as the SDK, CPU basic hardware.
  2. Upload this file and requirements.txt (Files tab -> Add file -> Upload).
  3. Wait for it to build (a few minutes the first time). Once it says
     "Running", copy the space name shown in the URL, e.g.
     huggingface.co/spaces/yourname/depth-anything-v2-small
     -> that's "yourname/depth-anything-v2-small".
  4. Paste that into REMOTE_SPACE in main.js.

Note: free Spaces go to sleep after a period of no traffic and take ~20-30s
to wake back up on the next request — the web app already accounts for this
with a timeout + automatic fallback to local inference.
"""

import gradio as gr
from transformers import pipeline

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)


def estimate_depth(image):
    # Returns a grayscale PIL Image the same size as the input — Gradio
    # serves it back to the caller as a URL the browser can load directly.
    result = depth_estimator(image)
    return result["depth"]


demo = gr.Interface(
    fn=estimate_depth,
    inputs=gr.Image(type="pil", label="Photo"),
    outputs=gr.Image(type="pil", label="Depth map"),
    title="Depth Anything V2 (Small) — depth estimation API",
    description="Server-side depth estimation for the spatial-photo web app.",
    api_name="predict",
)

if __name__ == "__main__":
    demo.queue().launch()
