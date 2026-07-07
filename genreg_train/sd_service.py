"""Images — text-to-image via Stable Diffusion 1.5 (diffusers), a separate
program that shares the Flask server, like diffuse_service / i2_service.

No training here — this just wires a pretrained SD1.5 checkpoint in as a
plain generation utility for the /images page. The pipeline is loaded lazily
(first request pays the cost) and cached as a process-wide singleton; a lock
serialises calls since one GPU can only run one generation at a time.
"""

import base64
import datetime
import io
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "runs", "images")

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # ungated HF mirror

LIM = {
    "steps": (1, 100),
    "guidance": (0.0, 20.0),
    "width": (64, 1024),
    "height": (64, 1024),
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class _Service:
    def __init__(self):
        self._pipe = None
        self._device = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._device = device

    @property
    def ready(self):
        return self._pipe is not None

    @property
    def device(self):
        return self._device

    def generate(self, prompt, negative_prompt="", steps=25, guidance=7.5,
                 seed=None, width=512, height=512):
        import torch

        with self._lock:
            self._load()
            steps = int(_clamp(steps, *LIM["steps"]))
            guidance = float(_clamp(guidance, *LIM["guidance"]))
            width = int(_clamp(width, *LIM["width"])) // 8 * 8
            height = int(_clamp(height, *LIM["height"])) // 8 * 8

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(int(seed))

            t0 = datetime.datetime.now()
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
            image = result.images[0]
            elapsed = (datetime.datetime.now() - t0).total_seconds()

            os.makedirs(OUT_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = os.path.join(OUT_DIR, f"{stamp}.png")
            image.save(path)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            return {
                "image_b64": b64,
                "path": os.path.relpath(path, ROOT),
                "elapsed": elapsed,
                "steps": steps,
                "guidance": guidance,
                "width": width,
                "height": height,
                "seed": seed,
                "device": self._device,
            }


HUB = _Service()
