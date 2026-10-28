"""video.py - the CogVideoX family on the image node: text-to-video, image-to-video and start+end frame interpolation, frames out as PNG + GIF / WebP / MP4.
Imported by main.py after its torch/diffusers patches; main.py owns the GPU lock, the loaded-model slot and the progress dict.

Model folders live in the transformer directory next to the Flux2 GGUF files, one folder per model in the diffusers layout every CogVideoX and CogVideoX-Fun repository uses:
  <name>/transformer/  <name>/vae/  <name>/scheduler/  <name>/text_encoder/  <name>/tokenizer/
What a folder can do is read from transformer/config.json, in_channels against the latent channel count C (out_channels, 16 for every released model):
  C      -> text-to-video (CogVideoX-2b / -5b / 1.5-5B)
  2C     -> image-to-video, start frame only (CogVideoX-5b-I2V / 1.5-5B-I2V); a folder named *control* / *pose* is a Fun control model and is listed but not run
  2C + 1 -> CogVideoX-Fun InP: text-to-video, start frame, or start AND end frame (the transformer sees a mask channel plus the encoded known frames)
The Fun InP sampler below is adapted from VideoX-Fun's CogVideoXFunInpaintPipeline (Apache-2.0, aigc-apps/VideoX-Fun) onto diffusers' own CogVideoX transformer, VAE and schedulers,
so no VideoX-Fun code or package is needed at runtime. Prompts are encoded with the model folder's own T5 encoder and cached by hash; nothing is downloaded.
"""
import os, gc, re, json, time, hashlib, subprocess, shutil
from pathlib import Path
from types import SimpleNamespace, MethodType
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel, AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXDPMScheduler, DDIMScheduler
from diffusers.models.upsampling import CogVideoXUpsample3D
from diffusers.utils.torch_utils import randn_tensor
from transformers import T5EncoderModel, T5Tokenizer

MODES = {"t2v": ["text_to_video"], "i2v": ["image_to_video"], "inp": ["text_to_video", "image_to_video", "start_end_to_video"]}
SCHEDULERS = {"ddim": DDIMScheduler, "cog_ddim": CogVideoXDDIMScheduler, "cog_dpm": CogVideoXDPMScheduler}
T5_DEVICE = os.getenv("VIDEO_T5_DEVICE", "cpu")        # cpu keeps the 9.5 GB encoder off the GPU side; it runs once per new prompt (results are cached)

class Interrupted(Exception): pass

# --- Model discovery ---

def _kind(cfg: dict, name: str) -> str:
    c = cfg.get("out_channels", 16)
    ch = cfg.get("in_channels", c)
    if ch == 2 * c + 1: return "inp"
    if ch == 2 * c: return "control" if re.search(r"control|pose", name, re.I) else "i2v"
    return "t2v" if ch == c else f"unknown ({ch} input channels)"

def _text_len(model_dir: Path) -> int: return json.loads((model_dir / "transformer" / "config.json").read_text()).get("max_text_seq_length", 226)   # the text length the transformer was trained on (226 for every released CogVideoX)

def frames_for(n: int, vt: int = 4, p_t=None) -> int:
    """Nearest valid frame count at or above n: 4k+1 for the VAE (vt = its temporal compression), and for CogVideoX 1.5 a latent length divisible by its temporal patch."""
    n = max(1, (int(n) - 1 + vt - 1) // vt * vt + 1)
    lf = (n - 1) // vt + 1
    return n + ((p_t - lf % p_t) * vt if p_t and lf % p_t else 0)

def _limits(cfg: dict, kind: str) -> dict:
    """What a model's position embeddings allow. A Fun InP model with absolute positions (2b-based) has a position table for sample_frames frames and no more (VideoX-Fun keeps the table's time axis);
    a model with learned positions (CogVideoX-5b-I2V) renders only at its trained size. Rotary models have neither limit."""
    rot = cfg.get("use_rotary_positional_embeddings")
    return {"max_frames": cfg.get("sample_frames", 49) if kind == "inp" and not rot else None, "fixed_size": [cfg.get("sample_width", 90) * 8, cfg.get("sample_height", 60) * 8] if cfg.get("use_learned_positional_embeddings") else None}

def _defaults(cfg: dict, kind: str) -> dict:
    """Starting values for the UI. Fun InP models are multi-resolution, so they start at the 512-class bucket (672x384), which is what an iGPU can turn around in minutes; the others start at their trained size."""
    p_t, lim = cfg.get("patch_size_t"), _limits(cfg, kind)
    w, h = (672, 384) if kind == "inp" and not lim["fixed_size"] else (cfg.get("sample_width", 90) * 8, cfg.get("sample_height", 60) * 8)
    n = (85 if kind == "inp" else 81) if p_t else 49
    return {"width": w, "height": h, "num_frames": min(n, lim["max_frames"]) if lim["max_frames"] else n, "fps": 16 if (p_t and kind != "inp") else 8, "steps": 50, "guidance_scale": 6.0, "scheduler": "ddim" if kind == "inp" else "cog_ddim"}

def video_models(root: Path) -> list:
    out = []
    for d in sorted(p for p in Path(root).iterdir() if (p / "transformer" / "config.json").is_file()):
        cfg = json.loads((d / "transformer" / "config.json").read_text())
        kind = _kind(cfg, d.name)
        out.append({"name": d.name, "family": "cogvideox", "kind": kind, "modes": MODES.get(kind, []), "defaults": _defaults(cfg, kind), **_limits(cfg, kind), "patch_size_t": cfg.get("patch_size_t"), "temporal_compression": cfg.get("temporal_compression_ratio", 4), "text_encoder": (d / "text_encoder").is_dir(), "note": "" if kind in MODES else "listed only - this node runs text-to-video, image-to-video and Fun InP models"})
    return out

# --- Text encoding (the model's own T5, cached) ---

def _mem_available() -> float:
    with open("/proc/meminfo") as f: return next(int(l.split()[1]) * 1024 for l in f if l.startswith("MemAvailable:"))

def text_embeds(model_dir: Path, prompt: str, negative: str, cache_dir: Path, free_gpu) -> tuple:
    """(prompt, negative) T5 embeddings, (1, text length, 4096) bf16 on cpu. Cached by encoder + length + text, so re-running a prompt (every segment of an animatic that shares one) never reloads T5.
    When the encoder would not fit beside what is loaded, free_gpu() unloads the denoiser first - it is reloaded on demand afterwards."""
    te, n = model_dir / "text_encoder", _text_len(model_dir)
    cache = cache_dir / f"""{hashlib.sha256(f"{te.resolve()}|{n}|{prompt}|{negative}".encode()).hexdigest()[:24]}.pt"""
    if cache.exists():
        d = torch.load(cache)
        return d["prompt"], d["negative"]
    if not te.is_dir(): raise FileNotFoundError(f"{te} is missing - the model folder needs its text_encoder/ and tokenizer/ (they ship in every CogVideoX repository)")
    need = sum(f.stat().st_size for f in te.iterdir() if f.suffix in (".safetensors", ".bin")) + 2e9
    if _mem_available() < need: free_gpu()
    tok = T5Tokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    enc = T5EncoderModel.from_pretrained(str(te), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).to(T5_DEVICE).eval()
    with torch.no_grad(): out = [enc(tok(t, padding="max_length", max_length=n, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids.to(T5_DEVICE))[0].to("cpu", torch.bfloat16) for t in (prompt, negative)]
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    torch.save({"prompt": out[0], "negative": out[1], "text": prompt, "negative_text": negative}, cache)
    return out[0], out[1]

# --- Images in ---

def fit(img: Image.Image, width: int, height: int) -> Image.Image:
    """Scale to cover width x height, then center-crop - keeps the aspect ratio instead of stretching a frame that was made at another shape."""
    s = max(width / img.width, height / img.height)
    img = img.convert("RGB").resize((max(width, round(img.width * s)), max(height, round(img.height * s))), Image.LANCZOS)
    l, t = (img.width - width) // 2, (img.height - height) // 2
    return img.crop((l, t, l + width, t + height))

def _tensor(img: Image.Image) -> torch.Tensor: return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)   # (3, H, W) in [0, 1]

# --- Fun InP helpers (after VideoX-Fun, Apache-2.0) ---

def _resize_mask(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Pixel mask (B,1,F,H,W) -> latent grid (B,1,f,h,w). The first frame resizes on its own because the CogVideoX VAE encodes frame 0 alone and every later 4 frames into one latent frame."""
    f, h, w = latent.shape[2:]
    first = F.interpolate(mask[:, :, :1], size=[1, h, w], mode="trilinear", align_corners=False)
    return first if f == 1 else torch.cat([first, F.interpolate(mask[:, :, 1:], size=[f - 1, h, w], mode="trilinear", align_corners=False)], dim=2)

def _fun_pos_embed(pe):
    """Fun InP models built on CogVideoX-2b use absolute 3D sin-cos positions and were trained at several resolutions with the base position table resized to each latent grid.
    diffusers rebuilds the table at the new size instead, which differs away from the base resolution - this swaps in the resize. Rotary (5b-based) models need nothing."""
    T, Hp, Wp = (pe.sample_frames - 1) // pe.temporal_compression_ratio + 1, pe.sample_height // pe.patch_size, pe.sample_width // pe.patch_size
    def forward(text_embeds, image_embeds):
        text_embeds = pe.text_proj(text_embeds)
        b, f, c, h, w = image_embeds.shape
        hp, wp = h // pe.patch_size, w // pe.patch_size
        x = pe.proj(image_embeds.reshape(-1, c, h, w)).view(b, f, -1, hp, wp).flatten(3).transpose(2, 3).flatten(1, 2)
        emb, t, base = torch.cat([text_embeds, x], dim=1).contiguous(), text_embeds.shape[1], pe.pos_embedding
        grid = F.interpolate(base[:, t:].view(1, T, Hp, Wp, -1).permute(0, 4, 1, 2, 3), size=[T, hp, wp], mode="trilinear", align_corners=False)
        pos = torch.cat([base[:, :t], grid.permute(0, 2, 3, 4, 1).reshape(1, -1, base.shape[-1])], dim=1)[:, : t + f * hp * wp]
        return emb + pos.to(emb.dtype)
    pe.forward = forward

def _up_fun(self, x: torch.Tensor) -> torch.Tensor:
    """CogVideoXUpsample3D (time-compressing) as VideoX-Fun's VAE runs it: the stream's first latent grows in space only, later chunks in time and space."""
    x = F.interpolate(x.squeeze(2), scale_factor=2.0)[:, :, None] if self._first else F.interpolate(x, scale_factor=2.0)
    b, c, t, h, w = x.shape
    return self.conv(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)).reshape(b, t, -1, h, w).permute(0, 2, 1, 3, 4)

def fun_decode(vae, z: torch.Tensor) -> torch.Tensor:
    """Decodes the way VideoX-Fun's VAE does for the models it trained: latent 0 alone (-> 1 frame), then pairs of latents (-> 8 frames), whatever the latent count's parity.
    diffusers' decode pairs from the start, which gives an even count no single first frame (22 latents -> 88 frames, not 85; Fun V1.5 runs on even counts) and groups
    the frames differently for the decoder's per-chunk normalization. Spatial tiling (with the VAE's own tile sizes and blending) keeps memory at diffusers' tiled level."""
    n, H, W = z.shape[2:]
    ups = [m for m in vae.decoder.modules() if isinstance(m, CogVideoXUpsample3D) and m.compress_time]
    def stream(t):
        out, cache = [], None
        for a, b in [(0, 1)] + [(s, min(s + 2, n)) for s in range(1, n, 2)]:
            for m in ups: m._first = a == 0
            x, cache = vae.decoder(vae.post_quant_conv(t[:, :, a:b]) if vae.post_quant_conv is not None else t[:, :, a:b], conv_cache=cache)
            out.append(x)
        return torch.cat(out, dim=2)
    for m in ups: m.forward = MethodType(_up_fun, m)
    try:
        th, tw = (vae.tile_latent_min_height, vae.tile_latent_min_width) if vae.use_tiling else (H, W)
        if H <= th and W <= tw: return stream(z)
        oh, ow = int(th * (1 - vae.tile_overlap_factor_height)), int(tw * (1 - vae.tile_overlap_factor_width))
        bh, bw = int(vae.tile_sample_min_height * vae.tile_overlap_factor_height), int(vae.tile_sample_min_width * vae.tile_overlap_factor_width)
        rows = [[stream(z[:, :, :, i:i + th, j:j + tw]) for j in range(0, W, ow)] for i in range(0, H, oh)]
        lines = []
        for i, row in enumerate(rows):
            line = []
            for j, tile in enumerate(row):
                if i > 0: tile = vae.blend_v(rows[i - 1][j], tile, bh)
                if j > 0: tile = vae.blend_h(row[j - 1], tile, bw)
                line.append(tile[:, :, :, :vae.tile_sample_min_height - bh, :vae.tile_sample_min_width - bw])
            lines.append(torch.cat(line, dim=4))
        return torch.cat(lines, dim=3)
    finally:
        for m in ups: del m.forward   # instance attribute off: the class method applies again

def _ref_noise(video: torch.Tensor, ratio: float, generator) -> torch.Tensor:
    """Fun V1.1+ InP adds a little noise to the known frames (add_noise_in_inpaint_model) - it trained that way, and it lets the motion move further from the reference."""
    noise = randn_tensor(video.shape, generator=generator, device=video.device, dtype=video.dtype) * ratio
    return video + torch.where(video == -1, torch.zeros_like(noise), noise)

# --- Loaded model ---

class VideoModel:
    def __init__(self, model_dir: Path, device, dtype=torch.bfloat16):
        cfg = json.loads((model_dir / "transformer" / "config.json").read_text())
        self.dir, self.name, self.cfg, self.kind, self.device, self.dtype = model_dir, model_dir.name, cfg, _kind(cfg, model_dir.name), device, dtype
        if self.kind not in MODES: raise ValueError(f"{self.name} is a {self.kind} model - this node runs text-to-video, image-to-video and Fun InP (start/end frame) models")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(str(model_dir), subfolder="transformer", torch_dtype=dtype, local_files_only=True).to(device)
        self.vae = AutoencoderKLCogVideoX.from_pretrained(str(model_dir), subfolder="vae", torch_dtype=dtype, local_files_only=True).to(device)
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.sched_cfg = json.loads((model_dir / "scheduler" / "scheduler_config.json").read_text())
        self.add_noise = bool(cfg.get("add_noise_in_inpaint_model", False))   # a Fun config key diffusers does not know - read from the file
        if self.kind == "inp" and not cfg.get("use_rotary_positional_embeddings"): _fun_pos_embed(self.transformer.patch_embed)

    def scheduler(self, name: str):
        if name not in SCHEDULERS: raise ValueError(f"""scheduler {name!r} - one of {", ".join(SCHEDULERS)}""")
        return SCHEDULERS[name].from_config(self.sched_cfg)

    def frames_for(self, n: int) -> int: return frames_for(n, self.vae.config.temporal_compression_ratio, self.transformer.config.patch_size_t)

    def generate(self, pe, ne, mode: str, start, end, width: int, height: int, num_frames: int, steps: int, guidance: float, seed: int, scheduler: str, dynamic_cfg: bool, noise_aug: float, on_step) -> list:
        """Returns the frames as PIL images. on_step(done, total) runs after every denoising step and may raise Interrupted."""
        if mode not in MODES[self.kind]: raise ValueError(f"""{self.name} ({self.kind}) does {", ".join(MODES[self.kind])} - not {mode}""")
        gen = torch.Generator(device=self.device).manual_seed(seed)
        pe, ne = pe.to(self.device, self.dtype), ne.to(self.device, self.dtype)
        if self.kind == "inp": return self._fun(pe, ne, start if mode != "text_to_video" else None, end if mode == "start_end_to_video" else None, width, height, self.frames_for(num_frames), steps, guidance, gen, self.scheduler(scheduler), dynamic_cfg, noise_aug, on_step)
        cls = CogVideoXImageToVideoPipeline if self.kind == "i2v" else CogVideoXPipeline
        pipe = cls(tokenizer=None, text_encoder=None, vae=self.vae, transformer=self.transformer, scheduler=self.scheduler(scheduler))
        def cb(p, i, t, kw): on_step(i + 1, steps); return kw
        kw = {"prompt_embeds": pe, "negative_prompt_embeds": ne, "width": width, "height": height, "num_frames": self.frames_for(num_frames), "num_inference_steps": steps, "guidance_scale": guidance, "use_dynamic_cfg": dynamic_cfg, "generator": gen, "output_type": "pil", "callback_on_step_end": cb}
        return pipe(**({**kw, "image": start} if self.kind == "i2v" else kw)).frames[0]

    def _fun(self, pe, ne, start, end, width, height, num_frames, steps, guidance, gen, sched, dynamic_cfg, noise_aug, on_step) -> list:
        sf, vs = self.vae.config.scaling_factor, 2 ** (len(self.vae.config.block_out_channels) - 1)
        lf = (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1
        cfg = guidance > 1.0
        emb = torch.cat([ne, pe]) if cfg else pe
        latents = randn_tensor((1, lf, self.vae.config.latent_channels, height // vs, width // vs), generator=gen, device=self.device, dtype=self.dtype) * sched.init_noise_sigma
        if start is None and end is None: inpaint = torch.zeros((1, lf, 1 + self.vae.config.latent_channels, height // vs, width // vs), device=self.device, dtype=self.dtype)
        else:
            video, mask = torch.zeros(1, 3, num_frames, height, width), torch.ones(1, 1, num_frames, height, width)   # mask 1 = generate, 0 = keep this pixel
            if start is not None: video[:, :, :] = _tensor(start)[None, :, None]; mask[:, :, 0] = 0
            if end is not None: video[:, :, -1] = _tensor(end); mask[:, :, -1] = 0
            masked = torch.where(mask < 0.5, video * 2 - 1, torch.full_like(video, -1.0)).to(self.device, self.dtype)   # known frames in [-1, 1], unknown pixels at -1
            if self.add_noise: masked = _ref_noise(masked, noise_aug, gen)
            with torch.no_grad(): mvl = self.vae.encode(masked)[0].mode() * sf
            ml = _resize_mask(1 - mask, mvl).to(self.device, self.dtype) * sf
            inpaint = torch.cat([ml, mvl], dim=1).permute(0, 2, 1, 3, 4)   # (1, f, 1 + 16, h, w): mask channel first, then the known-frame latents
        inpaint = torch.cat([inpaint] * 2) if cfg else inpaint
        sched.set_timesteps(steps, device=self.device)
        timesteps = sched.timesteps
        rope = CogVideoXPipeline._prepare_rotary_positional_embeddings(SimpleNamespace(vae_scale_factor_spatial=vs, transformer=self.transformer), height, width, lf, self.device) if self.transformer.config.use_rotary_positional_embeddings else None
        old = None
        for i, t in enumerate(timesteps):
            x = sched.scale_model_input(torch.cat([latents] * 2) if cfg else latents, t)
            with torch.no_grad(): noise = self.transformer(hidden_states=torch.cat([x, inpaint], dim=2), encoder_hidden_states=emb, timestep=t.expand(x.shape[0]), image_rotary_emb=rope, return_dict=False)[0].float()
            g = 1 + guidance * ((1 - np.cos(np.pi * ((steps - t.item()) / steps) ** 5.0)) / 2) if dynamic_cfg else guidance
            if cfg: u, c = noise.chunk(2); noise = u + g * (c - u)
            if isinstance(sched, CogVideoXDPMScheduler): latents, old = sched.step(noise, old, t, timesteps[i - 1] if i > 0 else None, latents, generator=gen, return_dict=False)
            else: latents = sched.step(noise, t, latents, generator=gen, return_dict=False)[0]
            latents = latents.to(self.dtype)
            on_step(i + 1, len(timesteps))
        with torch.no_grad(): video = fun_decode(self.vae, latents.permute(0, 2, 1, 3, 4) / sf)
        frames = (video / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 3, 0).float().cpu().numpy()
        return [Image.fromarray((f * 255).round().astype(np.uint8)) for f in frames]

# --- Frames out ---

def _gif(frames: list, path: Path, fps: float):
    """One palette for the whole clip (built from up to 8 frames spread through it), so colors do not shimmer frame to frame the way per-frame palettes do."""
    pick = frames[:: max(1, len(frames) // 8)][:8]
    mosaic = Image.new("RGB", (pick[0].width, pick[0].height * len(pick)))
    for i, f in enumerate(pick): mosaic.paste(f, (0, i * f.height))
    pal = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=[10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps)) for i in range(len(q))], loop=0, optimize=False, disposal=1)   # GIF delays are whole 1/100 s: alternating 120/130 ms keeps 8 fps exact on average

def _webp(frames: list, path: Path, fps: float): frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, quality=90, method=4)

def _mp4(frames: list, path: Path, fps: float):
    w, h = frames[0].size
    p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)], input=b"".join(f.convert("RGB").tobytes() for f in frames), capture_output=True)
    if p.returncode: raise RuntimeError(f"""ffmpeg: {p.stderr.decode(errors="replace")[-300:]}""")

WRITERS = {"gif": _gif, "webp": _webp, "mp4": _mp4}

def check_formats(formats: list):
    """Refuses unknown formats and a missing encoder before anything renders (a clip can take hours; failing at the save step would waste it)."""
    bad = [f for f in formats if f not in WRITERS]
    if bad or not formats: raise ValueError(f"""formats {bad or formats} - choose from {", ".join(WRITERS)}""")
    if "mp4" in formats and not shutil.which("ffmpeg"): raise ValueError("mp4 needs ffmpeg in the image node container (apt install ffmpeg) - gif and webp work without it")

def export(frames: list, out_dir: Path, stem: str, fps: float, formats: list, keep_frames: bool = True) -> dict:
    """Writes <stem>.<fmt> for each format, plus _video/<stem>/frame_NNNN.png (the lossless frames that joining clips and later edits work from). Returns paths relative to out_dir."""
    check_formats(formats)
    files = {}
    if keep_frames:
        d = out_dir / "_video" / stem
        d.mkdir(parents=True)
        for i, f in enumerate(frames): f.save(d / f"frame_{i:04d}.png")
        files["frames_dir"] = f"_video/{stem}"
    for fmt in formats:
        WRITERS[fmt](frames, out_dir / f"{stem}.{fmt}", fps)
        files[fmt] = f"{stem}.{fmt}"
    return files

def clip_frames(out_dir: Path, clip: str) -> list:
    """Frame files of a clip given as its frames folder (_video/<stem>) or any of its rendered files (<stem>.gif / .webp / .mp4)."""
    d = (out_dir / (clip.rstrip("/") if clip.startswith("_video/") else f"_video/{Path(clip).stem}")).resolve()
    if d.parent != (out_dir / "_video").resolve(): raise ValueError(f"clip {clip!r} is not a clip folder under _video/")
    files = sorted(d.glob("frame_*.png"))
    if not files: raise FileNotFoundError(f"no frames for clip {clip!r} at {d} - clips are joined from the frames each video run keeps (keep_frames)")
    return files

def join(out_dir: Path, clips: list, stem: str, fps: float, formats: list, drop_shared: bool = True) -> dict:
    """One clip from several, in order. drop_shared skips the first frame of every clip after the first: consecutive start/end segments share their keyframe, and showing it twice reads as a stutter."""
    frames = []
    for k, c in enumerate(clips): frames += [Image.open(f).convert("RGB") for f in clip_frames(out_dir, c)[(1 if k and drop_shared else 0):]]
    sizes = {f.size for f in frames}
    if len(sizes) > 1: raise ValueError(f"clips differ in size {sorted(sizes)} - render every segment at the same width and height")
    return {**export(frames, out_dir, stem, fps, formats), "frame_count": len(frames)}

