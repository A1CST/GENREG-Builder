"""Images — reverse direction: image/video -> the text prompt a diffusion
model would plausibly have used to generate that frame. A separate program
(pretrained, not evolved) sharing this server, like sd_service.

Pipeline (CLIP-Interrogator-style, kept small and self-contained):
  1. BLIP image captioning gives the base natural-language caption.
  2. CLIP ranks a handful of modifier banks (medium / style / lighting /
     quality) against the image; the best match per bank (if it clears a
     similarity floor) is appended, mirroring how SD prompts are usually
     written: "<subject>, <medium>, <style>, <lighting>, <quality tags>".

Video input is decoded to frames (imageio + bundled ffmpeg, same approach as
/api/pure/frames), each frame is interrogated independently, and everything
lands in a structured job folder:

  runs/images/reverse/<job_id>/
    manifest.json
    frames/frame_00001.png
    prompts/frame_00001.txt
    ...
"""

import datetime
import json
import os
import threading

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "runs", "images", "reverse")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

MODIFIER_BANKS = {
    "medium": [
        "a photograph", "a digital painting", "a 3d render", "an oil painting",
        "a watercolor painting", "a pencil sketch", "pixel art",
        "a comic book illustration", "a matte painting", "concept art",
        "a screenshot from a video game", "an anime illustration",
        "a charcoal drawing", "a vector illustration",
        "a black and white photograph",
    ],
    "style": [
        "cyberpunk", "steampunk", "fantasy art", "surrealism", "minimalist",
        "art nouveau", "art deco", "impressionist", "abstract",
        "photorealistic", "hyperrealistic", "low poly", "flat design",
        "retro", "vaporwave", "gothic", "baroque", "cubist",
    ],
    "lighting": [
        "studio lighting", "golden hour lighting", "dramatic lighting",
        "soft lighting", "neon lighting", "backlit", "volumetric lighting",
        "natural lighting", "cinematic lighting", "rim lighting",
    ],
    "quality": [
        "highly detailed", "4k", "8k", "sharp focus", "intricate details",
        "trending on artstation", "award winning", "professional photography",
        "octane render", "unreal engine render",
    ],
}
SIM_FLOOR = 0.17   # skip a bank if even its best match is this weak (raw CLIP
                    # cosine sims for these short modifier phrases typically
                    # land in the 0.17-0.25 range, not the 0.3+ people expect)

LIM = {
    "stride": (1, 100), "max_frames": (1, 300), "max_side": (128, 2048),
    "max_new_tokens": (5, 200), "top_k": (0, 4),
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class _Service:
    def __init__(self):
        self._blip_proc = None
        self._blip_model = None
        self._clip_proc = None
        self._clip_model = None
        self._bank_embeds = None   # {category: (phrases, tensor[N,D] normalised)}
        self._device = None
        self._lock = threading.Lock()

    def _load(self):
        if self._clip_model is not None:
            return
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._blip_proc = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_ID, torch_dtype=dtype).to(device)

        self._clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=dtype).to(device)

        self._device = device
        self._bank_embeds = {}
        with torch.no_grad():
            for cat, phrases in MODIFIER_BANKS.items():
                inputs = self._clip_proc(text=phrases, return_tensors="pt", padding=True).to(device)
                feats = self._clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self._bank_embeds[cat] = (phrases, feats)

    @property
    def ready(self):
        return self._clip_model is not None

    @property
    def device(self):
        return self._device

    def _caption(self, image, max_new_tokens):
        import torch
        # BLIP was trained on short (COCO-style) captions, so it emits EOS
        # and stops well short of max_new_tokens on its own by default —
        # min_new_tokens forces it to keep going, and sampling (instead of
        # greedy/beam decoding) + repetition penalty keep the forced extra
        # length from degenerating into repeated phrases.
        min_new_tokens = min(max_new_tokens, max(5, int(max_new_tokens * 0.7)))
        inputs = self._blip_proc(image, return_tensors="pt").to(
            self._device, self._blip_model.dtype)
        with torch.no_grad():
            out = self._blip_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        return self._blip_proc.decode(out[0], skip_special_tokens=True).strip()

    def _modifiers(self, image, top_k):
        if top_k <= 0:
            return {}
        import torch
        inputs = self._clip_proc(images=image, return_tensors="pt").to(
            self._device, self._clip_model.dtype)
        with torch.no_grad():
            img_feat = self._clip_model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            picks = {}
            for cat, (phrases, feats) in self._bank_embeds.items():
                sims = (img_feat @ feats.T).squeeze(0)
                k = min(top_k, sims.shape[0])
                top_vals, top_idx = sims.topk(k)
                hits = [{"phrase": phrases[int(i)], "score": float(v)}
                        for v, i in zip(top_vals.tolist(), top_idx.tolist()) if v >= SIM_FLOOR]
                if hits:
                    picks[cat] = hits
        return picks

    def interrogate(self, image, max_new_tokens=40, top_k=1):
        """image: PIL.Image (RGB) -> {caption, modifiers, prompt}.

        max_new_tokens controls how long the BLIP caption can run (detail of
        the "what's in the frame" part); top_k controls how many tags per
        modifier bank (medium/style/lighting/quality) get appended — 0 turns
        modifiers off entirely, so the prompt is just the caption."""
        max_new_tokens = int(_clamp(max_new_tokens, *LIM["max_new_tokens"]))
        top_k = int(_clamp(top_k, *LIM["top_k"]))
        with self._lock:
            self._load()
            image = image.convert("RGB")
            caption = self._caption(image, max_new_tokens)
            modifiers = self._modifiers(image, top_k)
            parts = [caption] + [h["phrase"] for hits in modifiers.values() for h in hits]
            prompt = ", ".join(parts)
            return {"caption": caption, "modifiers": modifiers, "prompt": prompt}

    # ---------------------------------------------------------------- jobs
    def new_job(self, kind, source_name):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        job_id = f"{stamp}-{kind}"
        job_dir = os.path.join(OUT_DIR, job_id)
        os.makedirs(os.path.join(job_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "prompts"), exist_ok=True)
        return job_id, job_dir, {
            "job_id": job_id, "kind": kind, "source": source_name,
            "created": stamp, "frames": [],
        }

    def _save_frame(self, job_dir, idx, image, max_side, max_new_tokens, top_k):
        w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        name = f"frame_{idx:05d}"
        img_path = os.path.join(job_dir, "frames", name + ".png")
        txt_path = os.path.join(job_dir, "prompts", name + ".txt")
        image.save(img_path)
        result = self.interrogate(image, max_new_tokens=max_new_tokens, top_k=top_k)
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(result["prompt"])
        return {
            "index": idx,
            "image": os.path.relpath(img_path, OUT_DIR).replace(os.sep, "/"),
            "prompt_file": os.path.relpath(txt_path, OUT_DIR).replace(os.sep, "/"),
            "caption": result["caption"],
            "modifiers": result["modifiers"],
            "prompt": result["prompt"],
        }

    def process_image_file(self, path, source_name, max_side=768, max_new_tokens=40, top_k=1):
        max_side = int(_clamp(max_side, *LIM["max_side"]))
        job_id, job_dir, manifest = self.new_job("image", source_name)
        image = Image.open(path)
        manifest["frames"].append(
            self._save_frame(job_dir, 1, image, max_side, max_new_tokens, top_k))
        manifest["params"] = {
            "max_side": max_side, "max_new_tokens": max_new_tokens, "top_k": top_k}
        self._write_manifest(job_dir, manifest)
        return manifest

    def process_video_file(self, path, source_name, stride=1, max_frames=30, max_side=768,
                            max_new_tokens=40, top_k=1):
        import imageio

        stride = int(_clamp(stride, *LIM["stride"]))
        max_frames = int(_clamp(max_frames, *LIM["max_frames"]))
        max_side = int(_clamp(max_side, *LIM["max_side"]))

        job_id, job_dir, manifest = self.new_job("video", source_name)
        reader = imageio.get_reader(path, "ffmpeg")
        try:
            kept = 0
            for i, frame in enumerate(reader):
                if i % stride != 0:
                    continue
                image = Image.fromarray(frame)
                kept += 1
                manifest["frames"].append(
                    self._save_frame(job_dir, kept, image, max_side, max_new_tokens, top_k))
                if kept >= max_frames:
                    break
        finally:
            reader.close()
        if not manifest["frames"]:
            raise RuntimeError("no frames decoded from this video")
        manifest["params"] = {
            "stride": stride, "max_frames": max_frames, "max_side": max_side,
            "max_new_tokens": max_new_tokens, "top_k": top_k,
        }
        self._write_manifest(job_dir, manifest)
        return manifest

    @staticmethod
    def _write_manifest(job_dir, manifest):
        with open(os.path.join(job_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)


HUB = _Service()
