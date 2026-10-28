"""
main.py - image node: Flux2 Klein (txt2img / img2img / inpaint / reference / sequences) and the CogVideoX family (text, image or start+end frames to video) on one GPU (FastAPI).

- One model is loaded at a time. A request naming another model unloads the current one first; a request naming none uses whatever image model is loaded (else the default).
- Outputs are never overwritten: a name already taken gets _2, _3 ... appended.
- Every generation runs in a worker thread under one GPU lock, so /system/status, /system/progress and /system/stop answer while a render runs (videos take minutes to hours on an iGPU).
- /system/stop interrupts at the next denoising step; the request then answers HTTP 409 "stopped".
- Video lives in video.py: model folders next to the Flux2 GGUF files, prompts encoded by each model's own T5 (cached under the embed vault), output as PNG frames + GIF / WebP / MP4.
- /models lists what the transformer folder holds and what each model can do; the portal builds its model selects from it.
"""

import os, sys, torch, gc, logging, threading, json, re, io, base64, time, math, asyncio, traceback, itertools
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from pathlib import Path
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
from types import ModuleType
import numpy as np
from PIL import Image

# =========================================================================
# PATCH 1: Metaclass-Driven Metamorphic Mock for 'flex_attention'
if not hasattr(torch.nn, "attention") or not hasattr(torch.nn.attention, "flex_attention"):
    if not hasattr(torch.nn, "attention"):
        attn_module = ModuleType("torch.nn.attention")
        sys.modules["torch.nn.attention"] = attn_module
        torch.nn.attention = attn_module

    class MetaMock(type):
        def __call__(cls, *args, **kwargs): return cls
        def __getattr__(cls, name): return type(name, (UniversalMock,), {})
        def __getitem__(cls, item): return cls
        def endswith(cls, *args, **kwargs): return False

    class UniversalMock(metaclass=MetaMock):
        def __call__(self, *args, **kwargs): return self
        def __getattr__(self, name): return type(name, (UniversalMock,), {})()
        def __getitem__(self, item): return self
        def endswith(self, *args, **kwargs): return False

    flex_mock = type("flex_attention", (UniversalMock,), {})
    sys.modules["torch.nn.attention.flex_attention"] = flex_mock
    torch.nn.attention.flex_attention = flex_mock

# PATCH 2: Temporary, Isolated Type Registry Hook
original_infer_schema = None
try:
    import torch._library.infer_schema
    original_infer_schema = torch._library.infer_schema.infer_schema

    def temporary_infer_schema(prototype_function, *args, **kwargs):
        try:
            if hasattr(prototype_function, "__annotations__"):
                eval_env = {
                    'torch': torch, 'Optional': Optional, 'tuple': tuple,
                    'list': list, 'Union': __import__('typing').Union,
                    'NoneType': type(None), 'str': str, 'int': int,
                    'float': float, 'bool': bool,
                    'Sequence': __import__('typing').Sequence,
                    'List': __import__('typing').List,
                }
                for param_name, current_val in list(prototype_function.__annotations__.items()):
                    if isinstance(current_val, str):
                        try:
                            prototype_function.__annotations__[param_name] = eval(current_val, eval_env)
                        except Exception:
                            pass
        except Exception:
            pass
        return original_infer_schema(prototype_function, *args, **kwargs)

    torch._library.infer_schema.infer_schema = temporary_infer_schema
except (ImportError, AttributeError):
    pass
# =========================================================================

import diffusers
import accelerate
from diffusers.quantizers.gguf import utils as gguf_utils

import diffusers.utils.import_utils
from safetensors.torch import load_file
from peft.utils import set_peft_model_state_dict

diffusers.utils.import_utils._accelerate_available = True
diffusers.utils.import_utils._accelerate_version = "1.1.0"

ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library but it was not found. pip install accelerate
"""
diffusers.utils.import_utils.BACKENDS_MAPPING["accelerate"] = (diffusers.utils.import_utils.is_accelerate_available, ACCELERATE_IMPORT_ERROR)

def patched_dtype(self): return torch.bfloat16
gguf_utils.GGUFParameter.dtype = property(patched_dtype)

import torch.nn.functional as F
original_sdpa = F.scaled_dot_product_attention
SDPA_CHUNK_ABOVE_GB = float(os.getenv("SDPA_CHUNK_ABOVE_GB", "8"))   # attention whose full score matrix would exceed this is computed in query chunks; 0 = never chunk
SDPA_CHUNK_GB = float(os.getenv("SDPA_CHUNK_GB", "1"))               # score-matrix size each chunk is cut to

def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """enable_gqa is dropped (this ROCm torch build rejects it). Long sequences - video attention runs 13k+ tokens, a 30 GB+ score matrix on the plain math kernel -
    are computed in query chunks: exact (attention rows are independent), with memory bounded by SDPA_CHUNK_GB whichever kernel torch picks."""
    gb = math.prod(query.shape[:-2]) * query.shape[-2] * key.shape[-2] * 4 / 1e9
    if SDPA_CHUNK_ABOVE_GB <= 0 or gb <= SDPA_CHUNK_ABOVE_GB or attn_mask is not None or is_causal: return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    step = max(1, int(query.shape[-2] * SDPA_CHUNK_GB / gb))
    return torch.cat([original_sdpa(query[..., i:i + step, :], key, value, dropout_p=dropout_p, scale=scale) for i in range(0, query.shape[-2], step)], dim=-2)

F.scaled_dot_product_attention = patched_sdpa
print("DEBUG: Global F.scaled_dot_product_attention patched.", flush=True)

sys.path.insert(0, os.path.abspath("local_overrides"))
from diffusers.quantizers.gguf import utils as utils  # noqa: F811
print(f"DEBUG: Using utils from {utils.__file__}", flush=True)

from accelerate.hooks import remove_hook_from_submodules
from pipeline_flux2_klein_inpaint import Flux2KleinInpaintPipeline
from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig
import peft  # noqa: F401
import video   # after the patches above: it imports diffusers' CogVideoX classes

# ---------------------------------------------------------------------------
app = FastAPI()
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")
LORA_DIR = os.getenv("LORA_DIR", "/app/models/lora/flux2")
LOG_FILE = os.path.join(OUTPUT_DIR, "logs.txt")
EMBED_VAULT_DIR = os.getenv("EMBED_VAULT_DIR", "/app/shared_vault/embeds")
JOBS_DIR = os.path.join(EMBED_VAULT_DIR, "jobs")
T5_CACHE_DIR = Path(EMBED_VAULT_DIR) / "t5"
TRANSFORMER_DIR = os.getenv("TRANSFORMER_DIR", "/app/models/transformers")
VAE_PATH = os.getenv("VAE_PATH", "/app/models/VAE")
DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "flux-2-klein-9b-Q6_K.gguf")
for _d in (OUTPUT_DIR, os.path.join(OUTPUT_DIR, "_video"), str(T5_CACHE_DIR)): os.makedirs(_d, exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai-image")
logger.addHandler(logging.StreamHandler())

GPU = threading.Lock()   # one generation at a time on the one GPU

# ---------------------------------------------------------------------------
# Progress tracking - written each denoising step, polled by the frontend
_progress: dict = {"step": 0, "total": 0, "job_id": "", "ts": 0.0, "status": "idle", "pct": 0}

def _on_step(job_id: str, done: int, total: int):
    """Records progress and honours /system/stop between steps."""
    _progress.update({"step": done, "total": total, "job_id": job_id, "ts": time.time(), "status": "running", "pct": int(done * 100 / max(total, 1))})
    logger.info(f"""[progress] job={job_id} step={done}/{total} ({_progress["pct"]}%)""")
    if engine.interrupt_flag.is_set(): raise video.Interrupted()

def _make_step_callback(job_id: str, total: int):
    def cb(pipe, step: int, timestep, kwargs: dict):
        _on_step(job_id, step + 1, getattr(pipe, "_num_timesteps", 0) or total)   # img2img / inpaint run only the strength share of the steps asked for
        return kwargs
    return cb

# -----------------------------------------------------------
class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: Optional[str] = ""
    embed_job_id: Optional[str] = None
    image_path: Optional[str] = None
    image: Optional[str] = None
    mask_image: Optional[str] = None   # data URI OR path relative to OUTPUT_DIR
    reference_image: Optional[str] = None   # data URI OR path relative to OUTPUT_DIR - conditions the repainted area on this picture
    strength: Optional[float] = 0.75
    width: int = 1600                  # 0 with an init image = taken from it (both 0 = its own size, one 0 = that side from its aspect ratio)
    height: int = 900
    steps: int = 4
    guidance_scale: float = 1.0
    shift: Optional[float] = None
    seed: int = -1
    model_path: Optional[str] = None   # None = keep whatever image model is loaded (else DEFAULT_IMAGE_MODEL); naming another one swaps it in
    vae_path: str = "flux2"
    loras: list = []
    vae_tiling: bool = True
    offload_mode: str = "none"   # "none" | "vae_cpu" | "sequential"
    meta: list = ["prompt", "width", "height", "guidance_scale", "steps", "seed", "loras"]
    sequence_dir: Optional[str] = None

class VideoRequest(BaseModel):
    """Unset numbers take the model's own defaults (see /models). mode is inferred from the images given: none -> text_to_video, start -> image_to_video, start + end -> start_end_to_video."""
    model: Optional[str] = None                 # model folder name under the transformer dir; None = the loaded video model when it can do the mode, else the first that can
    prompt: str = ""                            # may be blank when start/end images carry the content (an animatic gap with no description)
    negative_prompt: str = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."
    mode: Optional[str] = None
    start_image: Optional[str] = None           # data URI or a path inside the outputs folder (e.g. a keyframe the image model just made)
    end_image: Optional[str] = None
    width: Optional[int] = None                 # both unset on a Fun InP model with a start frame = the start frame's shape at the model's default pixel count
    height: Optional[int] = None
    num_frames: Optional[int] = None
    fps: Optional[float] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    scheduler: Optional[str] = None             # ddim | cog_ddim | cog_dpm
    dynamic_cfg: bool = False
    noise_aug_strength: float = 0.0563          # noise on the known frames for Fun V1.1+ InP models (what they trained with)
    seed: int = -1
    formats: List[str] = ["gif"]                # any of gif, webp, mp4
    keep_frames: bool = True                    # PNG frames under _video/<name>/ - joining clips works from these
    offload_mode: str = "none"                  # "none" | "sequential" (transformer blocks visit the GPU one at a time)
    name: Optional[str] = None                  # output file prefix

class JoinRequest(BaseModel):
    clips: List[str]                            # clip frame folders (_video/<name>) or any of their rendered files, in order
    fps: float = 8
    formats: List[str] = ["gif"]
    drop_shared: bool = True                    # skip each later clip's first frame - consecutive start/end segments share their keyframe
    name: Optional[str] = None

class SilentlyEmptyDict(dict):
    def __len__(self): return 0
    def __bool__(self): return False
# ------------------------------------------------------------

class APUEngine:
    def __init__(self):
        self.pipe = None
        self.is_loaded = False
        self.model_path: Optional[str] = None
        self.family: Optional[str] = None   # "flux" or "cogvideox"
        self.video: Optional[video.VideoModel] = None
        self.interrupt_flag = threading.Event()
        self.active_loras: list = []
        self.last_error: Optional[str] = None
        self._base_scheduler_config = None
        self._lora_hooks: list = []
        self._last_lora_config: list = []
        self.expected_text_dim: int = int(os.getenv("EXPECTED_TEXT_DIM", str(4096 * 3)))   # Flux2 Klein: Qwen3 layers 9/18/27 concatenated

    def load(self, model_path: str, vae_path: str = "flux2"):
        """Loads model_path (a .gguf -> Flux2 Klein, a folder -> CogVideoX family), unloading whatever else is loaded first."""
        if self.is_loaded and self.model_path == model_path: return
        full_model_path = os.path.join(TRANSFORMER_DIR, model_path)
        if not os.path.exists(full_model_path): raise FileNotFoundError(f"{model_path} is not in the transformer folder - see /models")   # checked before unloading, so a wrong name keeps the loaded model
        if self.is_loaded: self.unload()
        cuda_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        if model_path.endswith(".gguf"):
            self._load_flux(full_model_path, vae_path, cuda_dev)
            logger.info("Stripping accelerate hooks from transformer...")
            remove_hook_from_submodules(self.pipe.transformer)
            logger.info(f"Pinning transformer and VAE to {cuda_dev}...")
            self.pipe.transformer.to(cuda_dev)
            self.pipe.vae.to(cuda_dev)
            logger.info(f"Pipeline execution device: {self.pipe._execution_device}")
            self.pipe._original_state_dict = SilentlyEmptyDict()
            self.pipe.transformer._original_state_dict = SilentlyEmptyDict()
            self.pipe.vae._original_state_dict = SilentlyEmptyDict()
            self.set_vae_tiling(True)
            self.family = "flux"
        else:
            self.video, self.family = video.VideoModel(Path(full_model_path), cuda_dev, torch.bfloat16), "cogvideox"
        self.model_path, self.is_loaded = model_path, True
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Engine load complete: {model_path} ({self.family}) in {time.time() - t0:.1f}s - {_mem_info_str()}")

    def ensure_image(self, model_path: Optional[str], vae_path: str):
        """None keeps a loaded image model; with a video model (or nothing) loaded it means the default image model."""
        self.load(model_path or (self.model_path if self.family == "flux" else DEFAULT_IMAGE_MODEL), vae_path)

    def _load_flux(self, model_path, vae_path, cuda_dev):
        vae_path_dir = os.path.join(VAE_PATH, vae_path)
        logger.info(f"Loading Flux VAE from {vae_path_dir}, transformer from {model_path}")
        with open(os.path.join(vae_path_dir, "config.json")) as f:
            vae_config = json.load(f)
        vae = AutoencoderKLFlux2.from_config(vae_config)
        vae.load_state_dict(load_file(os.path.join(vae_path_dir, "flux2-vae.safetensors")))
        vae.to(torch.bfloat16)
        transformer = Flux2Transformer2DModel.from_single_file(model_path, config=os.path.join(os.path.dirname(model_path), "config.json"), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)

        class GGUFPatchedTransformer(transformer.__class__):
            @property
            def dtype(self): return torch.bfloat16
        transformer.__class__ = GGUFPatchedTransformer

        scheduler_config = {"num_train_timesteps": 1000, "shift": 1.0}
        self._base_scheduler_config = scheduler_config
        self.pipe = Flux2KleinInpaintPipeline(transformer=transformer, vae=vae, text_encoder=None, tokenizer=None, scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config), is_distilled=True)

    def execution_device(self) -> torch.device:
        if self.family == "cogvideox": return torch.device(self.video.device)
        if self.pipe is None: return torch.device("cpu")
        dev = self.pipe._execution_device
        if torch.cuda.is_available() and torch.device(dev).type == "cpu": logger.warning(f"cuda available but pipe._execution_device={dev}. Unload and reload to fix.")
        return torch.device(dev)

    def resolve_embed_path(self, request: "GenerationRequest"):
        job_dir = os.path.join(JOBS_DIR, request.embed_job_id)
        embeds_path = os.path.join(job_dir, "embeds.npy")
        meta_path = os.path.join(job_dir, "meta.json")
        if not os.path.exists(embeds_path): raise FileNotFoundError(f"No embeds.npy for embed_job_id={request.embed_job_id} at {embeds_path}")
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return embeds_path, meta

    def load_local_embeddings(self, request: "GenerationRequest") -> torch.Tensor:
        embeds_path, meta = self.resolve_embed_path(request)
        logger.info(f"Loading embeddings from: {embeds_path}")
        np_arr = np.load(embeds_path)
        logger.info(f"Loaded array shape={np_arr.shape} dtype={np_arr.dtype}")
        if np_arr.ndim != 3 or np_arr.shape[0] != 1: raise ValueError(f"Expected (1, seq_len, dim), got {np_arr.shape}")
        if np_arr.shape[-1] != self.expected_text_dim: raise ValueError(f"Embeds last dim {np_arr.shape[-1]} != expected {self.expected_text_dim} for {self.family}.")
        if not np.isfinite(np_arr).all(): raise ValueError("Embeds contain NaN/Inf - re-encode this prompt.")
        if meta and meta.get("stats", {}).get("std", 1.0) < 1e-6: logger.warning(f"""Near-zero std in embeddings: {meta.get("stats")}""")
        return torch.from_numpy(np_arr).to(device=self.execution_device(), dtype=torch.bfloat16)

    # -- LoRA helpers (unchanged from original) --
    def clear_loras(self):
        removed = 0
        for handle in self._lora_hooks:
            try:
                handle.remove()
                removed += 1
            except Exception:
                pass
        self._lora_hooks.clear()
        if removed: logger.info(f"Removed {removed} LoRA forward hooks.")
        self.active_loras = []
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def apply_lora_weights(self, loras_list: list):
        normalized = [{"path": l.get("path", ""), "scale": round(float(l.get("scale", 1.0)), 4)} for l in loras_list if l.get("path")]
        if normalized == self._last_lora_config:
            logger.info(f"[LoRA] config unchanged, skipping re-apply")
            return
        self.clear_loras()
        if not normalized or self.pipe is None:
            self._last_lora_config = []
            return
        model = self.pipe.transformer
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        module_index = {name: mod for name, mod in model.named_modules() if hasattr(mod, "weight") and mod.weight is not None}
        for i, lora_cfg in enumerate(loras_list[:3]):
            raw_path = lora_cfg.get("path", "")
            if not raw_path: continue
            path = os.path.join(LORA_DIR, raw_path)
            scale = float(lora_cfg.get("scale", 1.0))
            name = lora_cfg.get("name") or f"lora_{i}_{os.path.basename(raw_path)}"
            try:
                self._apply_single_lora(path, name, scale, module_index, dev, dtype)
            except Exception as e:
                logger.error(f"[LoRA] Failed to apply {name}: {e}\n{traceback.format_exc()}")
        self._last_lora_config = normalized

    def _register_lora_hook(self, module, lora_down, lora_up, eff_scale, tag, resolved_name):
        import torch.nn.functional as F_local
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None: out_features, in_features = module.weight.shape[0], module.weight.shape[1]

        if lora_down.shape[1] == in_features and lora_up.shape[0] == out_features:
            _ld, _lu, _s = lora_down, lora_up, eff_scale

            def _hook_lora(mod, inp, out, ld=_ld, lu=_lu, s=_s):
                x = inp[0]
                if ld.device != x.device or ld.dtype != x.dtype:
                    ld = ld.to(x.device, x.dtype)
                    lu = lu.to(x.device, x.dtype)
                orig_shape = out.shape
                x2 = x.reshape(-1, x.shape[-1])
                delta = F_local.linear(F_local.linear(x2, ld), lu)
                return out + s * delta.view(orig_shape)

            self._lora_hooks.append(module.register_forward_hook(_hook_lora))
            return True

        A, B = lora_down, lora_up
        m, n = A.shape
        p, q = B.shape
        if m * p == out_features and n * q == in_features:
            _A, _B, _s, _m, _n, _p, _q = A, B, eff_scale, m, n, p, q

            def _hook_lokr(mod, inp, out, A=_A, B=_B, s=_s, m=_m, n=_n, p=_p, q=_q):
                x = inp[0]
                if A.device != x.device or A.dtype != x.dtype:
                    A = A.to(x.device, x.dtype)
                    B = B.to(x.device, x.dtype)
                orig_shape = out.shape
                x_r = x.reshape(*x.shape[:-1], n, q)
                temp = torch.einsum('ij,...jl->...il', A, x_r)
                y = torch.einsum('kl,...il->...ik', B, temp)
                return out + s * y.reshape(*orig_shape[:-1], m * p)

            self._lora_hooks.append(module.register_forward_hook(_hook_lokr))
            logger.info(f"""[LoRA] {tag}: '{resolved_name}' LoKR A={tuple(A.shape)} B={tuple(B.shape)}""")
            return True

        logger.warning(f"""[LoRA] {tag}: shape mismatch '{resolved_name}' in={in_features} out={out_features} down={tuple(lora_down.shape)} up={tuple(lora_up.shape)}""")
        return False

    def _resolve_fused_target(self, module_index, block_prefix, lora_up, lora_down):
        total_out = lora_up.shape[0]
        in_dim = lora_down.shape[1]
        candidates = [(n, m) for n, m in module_index.items() if n.startswith(block_prefix + ".") and hasattr(m, "out_features") and hasattr(m, "in_features")]
        if not candidates: return None
        for n, m in candidates:
            if m.out_features == total_out and m.in_features == in_dim: return [(m, n, slice(0, total_out))]
        same_in = [(n, m) for n, m in candidates if m.in_features == in_dim]
        running, chosen = 0, []
        for n, m in same_in:
            if running >= total_out: break
            chosen.append((n, m, running))
            running += m.out_features
        if running == total_out and chosen: return [(m, n, slice(off, off + m.out_features)) for n, m, off in chosen]
        return None

    def _apply_single_lora(self, path, name, scale, module_index, dev, dtype):
        state_dict = load_file(path, device="cpu")
        all_keys = sorted(state_dict.keys())
        logger.info(f"[LoRA] {name} - {len(all_keys)} tensors. First 5: {all_keys[:5]}")

        down_map, up_map, alpha_map = {}, {}, {}
        for k, v in state_dict.items():
            if k.endswith(".alpha"):
                alpha_map[k[:-len(".alpha")]] = float(v.item())
                continue
            is_weight = k.endswith(".weight")
            stem = k[:-7] if is_weight else k
            if stem.endswith(".lora_down") or stem.endswith(".lora_A") or stem.endswith(".down"):
                down_map[stem.rsplit(".", 1)[0]] = v
            elif stem.endswith(".lora_up") or stem.endswith(".lora_B") or stem.endswith(".up"):
                up_map[stem.rsplit(".", 1)[0]] = v

        matched, skipped_keys = 0, []
        for base in down_map:
            if base not in up_map: continue
            lora_down = down_map[base].to(dev, dtype)
            lora_up = up_map[base].to(dev, dtype)
            rank = lora_down.shape[0]
            alpha = alpha_map.get(base, float(rank))
            eff_scale = scale * alpha / rank

            if base.endswith("img_attn.qkv") or base.endswith("txt_attn.qkv"):
                is_txt = base.endswith("txt_attn.qkv")
                block_prefix = base.rsplit(".", 2)[0]
                for pfx in ("diffusion_model.", "transformer.", "base_model.model."):
                    if block_prefix.startswith(pfx): block_prefix = block_prefix[len(pfx):]
                block_prefix = (block_prefix.replace("double_blocks", "transformer_blocks").replace("single_blocks", "single_transformer_blocks"))
                targets = ["add_q_proj", "add_k_proj", "add_v_proj"] if is_txt else ["to_q", "to_k", "to_v"]
                out_dim = lora_up.shape[0]
                if out_dim % 3 != 0:
                    skipped_keys.append(base)
                    continue
                chunk = out_dim // 3
                sub_applied = 0
                for idx, tgt in enumerate(targets):
                    resolved_key = f"{block_prefix}.attn.{tgt}"
                    module = module_index.get(resolved_key)
                    if module is None: continue
                    up_slice = lora_up[idx * chunk:(idx + 1) * chunk, :]
                    if self._register_lora_hook(module, lora_down, up_slice, eff_scale, name, resolved_key): sub_applied += 1
                if sub_applied == 0:
                    skipped_keys.append(base)
                else:
                    matched += 1
                continue

            module, resolved = self._find_module(module_index, base)
            if module is not None:
                if self._register_lora_hook(module, lora_down, lora_up, eff_scale, name, resolved):
                    matched += 1
                else:
                    skipped_keys.append(base)
                continue

            block_m = re.search(r'^(.*?\.\d+)\.', base)
            if block_m:
                fused = self._resolve_fused_target(module_index, block_m.group(1), lora_up, lora_down)
                if fused:
                    sub_applied = 0
                    for sub_mod, sub_name, sl in fused:
                        up_slice = lora_up[sl, :]
                        if self._register_lora_hook(sub_mod, lora_down, up_slice, eff_scale, name, sub_name): sub_applied += 1
                    if sub_applied:
                        matched += 1
                        logger.info(f"""[LoRA] {name}: '{base}' fused split -> {[s for _,s,_ in fused]}""")
                        continue
            skipped_keys.append(base)

        if skipped_keys: logger.warning(f"[LoRA] {name}: {len(skipped_keys)} unmatched. First: {skipped_keys[:5]}")
        if matched:
            self.active_loras.append(name)
            logger.info(f"[LoRA] {name}: {matched}/{len(down_map)} adapters applied")
        else:
            logger.error(f"[LoRA] {name}: ZERO adapters matched - check key format.")

    def _find_module(self, module_index, key):
        for prefix in ["diffusion_model.", "transformer.", "base_model.model."]:
            if key.startswith(prefix): key = key[len(prefix):]
        name_swaps = {"double_blocks": "transformer_blocks",
                      "single_blocks": "single_transformer_blocks",
                      "img_attn.proj": "attn.to_out.0",
                      "txt_attn.proj": "attn.to_add_out",
                      "img_mlp.0": "ff.net.0.proj",
                      "img_mlp.2": "ff.net.2",
                      "txt_mlp.0": "ff_context.net.0.proj",
                      "txt_mlp.2": "ff_context.net.2",
                      "linear1": "proj_mlp",
                      "linear2": "proj_out",
                      "img_in": "x_embedder",
                      "final_layer.linear": "proj_out",
                      "final_layer.adaLN_modulation.1": "norm_out.linear",
                      "double_stream_modulation_img.lin": "time_text_embed.timestep_embedder.linear_2",
                      "double_stream_modulation_txt.lin": "time_text_embed.text_embedder.linear_2"}

        mapped_key = key
        for old, new in name_swaps.items():
            mapped_key = mapped_key.replace(old, new)
        if mapped_key in module_index: return module_index[mapped_key], mapped_key
        alt_key = mapped_key.replace("net.0.proj", "linear_in").replace("net.2", "linear_out")
        if alt_key in module_index: return module_index[alt_key], alt_key
        block_match = re.search(r'\.(\d+)\.', mapped_key)
        block_id = block_match.group(0) if block_match else None
        key_norm = mapped_key.replace("_", ".").lower()
        candidates = []
        for m_name, m_obj in module_index.items():
            if block_id and block_id not in m_name: continue
            m_norm = m_name.replace("_", ".").lower()
            if m_norm.endswith(key_norm) or key_norm.endswith(m_norm): candidates.append((m_name, m_obj))
        if len(candidates) == 1: return candidates[0][1], candidates[0][0]
        return None, key

    def set_vae_tiling(self, enabled: bool):
        try:
            vae = getattr(self.pipe, "vae", None) if self.pipe else None
            if vae is None: return
            if enabled and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            elif not enabled and hasattr(vae, "disable_tiling"):
                vae.disable_tiling()
        except Exception as e:
            logger.warning(f"VAE tiling toggle failed (non-fatal): {e}")

    def maybe_override_scheduler_shift(self, shift: Optional[float]):
        if shift is None or self.pipe is None or self._base_scheduler_config is None: return
        try:
            cfg = dict(self._base_scheduler_config)
            cfg["shift"] = shift
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(cfg)
            logger.info(f"Scheduler shift overridden to {shift}")
        except Exception as e:
            logger.warning(f"Failed to override scheduler shift={shift}: {e}")

    def interrupt(self):
        logger.info("Interrupt signal sent...")
        self.interrupt_flag.set()

    def unload(self):
        logger.info(f"Unloading {self.model_path} and clearing VRAM...")
        self.clear_loras()
        self._last_lora_config = []
        self.pipe, self.video, self.model_path, self.family, self.is_loaded = None, None, None, None, False
        gc.collect()
        torch.cuda.empty_cache()

engine = APUEngine()

def _mem_info_str() -> str:
    """
    Note: the reported "total capacity" for an APU/iGPU is a driver-level GTT/VRAM split, not something torch or this process can change.
    The two real levers are (a) the host kernel param `amdgpu.gttsize=<MB>` (needs a reboot) or (b) the BIOS UMA framebuffer size - not fixable from Python.
    PYTORCH_ALLOC_CONF=expandable_segments:True (already set) only reduces fragmentation within whatever ceiling the driver reports.
    """
    if not torch.cuda.is_available(): return "cuda not available"
    try:
        free, total = torch.cuda.mem_get_info()
        return f"{free/1e9:.2f} GB free / {total/1e9:.2f} GB total"
    except Exception as e:
        return f"mem_get_info failed: {e}"

class _VaeDeviceGuard:
    """
    Wraps vae.encode AND vae.decode so the VAE only lives on GPU for the duration of each call.
    Fixes the dtype/device mismatch that occurred when only decode() was special-cased - encode() runs earlier, during prepare_latents/prepare_image_latents, while image tensors are already on CUDA.
    """
    def __init__(self, vae, device):
        self.vae = vae
        self.device = device
        self._orig_encode = vae.encode
        self._orig_decode = vae.decode

    def __enter__(self):
        def _wrap(fn):
            def wrapped(sample, *args, **kwargs):
                self.vae.to(self.device)
                try:
                    return fn(sample, *args, **kwargs)
                finally:
                    self.vae.to("cpu")
                    torch.cuda.empty_cache()
            return wrapped
        self.vae.encode = _wrap(self._orig_encode)
        self.vae.decode = _wrap(self._orig_decode)
        self.vae.to("cpu")
        logger.info(f"[vae-cpu-offload] enabled - {_mem_info_str()}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.vae.encode = self._orig_encode
        self.vae.decode = self._orig_decode
        self.vae.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[vae-cpu-offload] disabled - {_mem_info_str()}")
        return False

class _SequentialBlockOffload:
    """
    Experimental "wait-overnight" mode.
    Moves each transformer block to GPU immediately before its forward pass and back to CPU immediately after, using plain PyTorch hooks - deliberately NOT accelerate's cpu_offload/AlignDevicesHook, which conflicts with this GGUF-quantized transformer's own .to() calls during forward (see the comment in APUEngine.load()).
    Much slower, frees nearly all transformer VRAM between blocks.
    """
    def __init__(self, transformer, device):
        self.transformer = transformer
        self.device = device
        self.handles = []
        self.block_lists = [getattr(transformer, attr) for attr in ("transformer_blocks", "single_transformer_blocks") if getattr(transformer, attr, None) is not None]

    def __enter__(self):
        def _pre(mod, inp):
            mod.to(self.device)
            return inp

        def _post(mod, inp, out):
            mod.to("cpu")
            return out

        for blocks in self.block_lists:
            for block in blocks:
                self.handles.append(block.register_forward_pre_hook(_pre))
                self.handles.append(block.register_forward_hook(_post))
                block.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[sequential-offload] enabled - {_mem_info_str()}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        for blocks in self.block_lists:
            for block in blocks:
                block.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[sequential-offload] disabled - {_mem_info_str()}")
        return False

def _offload_guards(mode: str, transformer, vae, device) -> list:
    """Context managers for an offload mode, entered before the pipeline call and exited after it (whatever happens)."""
    if mode not in ("none", "vae_cpu", "sequential"): raise HTTPException(400, f"offload_mode {mode!r} - one of none, vae_cpu, sequential")
    return ([_VaeDeviceGuard(vae, device)] if mode == "vae_cpu" and vae is not None else []) + ([_SequentialBlockOffload(transformer, device)] if mode == "sequential" else [])

# --- Helpers ---

def _out_path(rel: str) -> str:
    """A path inside OUTPUT_DIR from a relative one (subfolders allowed); anything that would leave the folder is refused."""
    p = os.path.realpath(os.path.join(OUTPUT_DIR, rel.lstrip("/")))
    if not p.startswith(os.path.realpath(OUTPUT_DIR) + os.sep): raise HTTPException(400, f"path {rel!r} is outside the outputs folder")
    if not os.path.exists(p): raise HTTPException(400, f"not found in outputs: {rel}")
    return p

def _image_in(field: str) -> Image.Image:
    """An input picture given as a data URI / bare base64, or as a path inside the outputs folder (what the portal passes between pipeline steps)."""
    return decode_b64_image(field) if field.startswith("data:") or len(field) > 512 else Image.open(_out_path(field)).convert("RGB")

def _load_mask_image(mask_field: str, width: int, height: int) -> Image.Image:
    """
    Load mask image from either a data URI or a path relative to OUTPUT_DIR.
    Returns a grayscale PIL image resized to (width, height).
    White = inpaint, Black = keep.
    """
    if not mask_field: return Image.new("L", (width, height), 255)  # all white = full inpaint
    return _image_in(mask_field).resize((width, height)).convert("L")

def decode_b64_image(b64_str: str) -> Image.Image:
    if "base64," in b64_str: b64_str = b64_str.split("base64,")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def _seed(s: Optional[int]) -> int: return torch.randint(0, 2**32 - 1, (1,)).item() if s is None or s < 0 else s

_UNSAFE = re.compile(r"[^\w\-]")   # anything but letters, digits, _ and - is replaced in names that become file or folder names

def _free(stem: str, taken) -> str:
    """stem, else stem_2, stem_3 ... - the first name taken(name) reports unused, so an output never overwrites an earlier one."""
    return next(s for s in (stem if i == 1 else f"{stem}_{i}" for i in itertools.count(1)) if not taken(s))

def _free_file(folder: str, stem: str, ext: str) -> str: return f"{_free(stem, lambda s: os.path.exists(os.path.join(folder, s + ext)))}{ext}"

def _clip_taken(stem: str) -> bool: return os.path.exists(os.path.join(OUTPUT_DIR, "_video", stem)) or any(os.path.exists(os.path.join(OUTPUT_DIR, f"{stem}.{f}")) for f in video.WRITERS)

def _stem(prefix: Optional[str], seed: int) -> str: return _free(f"""{_UNSAFE.sub("_", prefix or "vid")[:40]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{seed}""", _clip_taken)

def _fit_size(w: int, h: int, img: Image.Image) -> tuple:
    """Width/height 0 taken from an init image: both 0 = its own size, one 0 = that side from its aspect ratio. Rounded down to multiples of 16 (the latent grid)."""
    if not (w or h): w, h = img.width, img.height
    elif not h: h = img.height * w / img.width
    elif not w: w = img.width * h / img.height
    return max(16, int(w) // 16 * 16), max(16, int(h) // 16 * 16)

def _locked(fn, *args):
    """Runs one generation on the GPU: waits its turn, starts with a clear stop flag, and leaves progress idle (or stopped / error) when done."""
    with GPU:
        engine.interrupt_flag.clear()
        try: return fn(*args)
        except video.Interrupted:
            _progress.update({"status": "stopped"})
            raise HTTPException(409, "stopped by request")
        except torch.cuda.OutOfMemoryError:
            logging.error(f"OOM during generation:\n{traceback.format_exc()}")
            engine.last_error = f"HIP/CUDA out of memory. {_mem_info_str()}. Try a smaller size, a heavier offload_mode, fewer frames or no LoRAs."
            _progress.update({"status": "error"})
            raise HTTPException(500, engine.last_error)
        except HTTPException: raise
        except Exception as e:
            logging.error(f"""{getattr(fn, "__name__", fn)} failed:\n{traceback.format_exc()}""")
            engine.last_error = str(e)
            _progress.update({"status": "error"})
            raise HTTPException(500, f"Pipeline failed: {e}")
        finally:
            if _progress.get("status") in ("running", "stopping"): _progress.update({"status": "idle"})   # stopping: the stop came after the last step, so the run finished anyway
            gc.collect()
            torch.cuda.empty_cache()

async def _gpu(fn, *args): return await asyncio.to_thread(_locked, fn, *args)

# --- System endpoints ---

@app.post("/system/load")
async def load_system(model_path: str = DEFAULT_IMAGE_MODEL, vae_path: str = "flux2"):
    """Loads model_path (switching from whatever is loaded). Answers at once while a generation holds the GPU instead of queueing behind it."""
    if GPU.locked(): return {"status": "Error", "detail": "busy - a generation is running (stop it or wait for it)"}
    try: await asyncio.to_thread(_locked, engine.load, model_path, vae_path)
    except HTTPException as e: return {"status": "Error", "detail": e.detail}   # traceback is in the node log
    return {"status": "Loaded into memory", "model": engine.model_path, "family": engine.family}

@app.post("/system/stop")
def stop_generation():
    """Stops the running generation at its next denoising step (its request answers 409; a sequence keeps the frames it finished). Jobs queued behind it still run."""
    if not GPU.locked(): return {"status": "idle - nothing to stop"}
    engine.interrupt()
    _progress.update({"status": "stopping"})
    return {"status": "Interrupting..."}

@app.post("/system/clear_error")
def clear_error():
    engine.last_error = None
    engine.interrupt_flag.clear()
    _progress.update({"status": "idle", "step": 0, "total": 0, "pct": 0})
    return {"status": "ok"}

@app.post("/system/unload")
async def unload_system():
    engine.interrupt()   # a running generation stops at its next step, then the unload takes the GPU
    await asyncio.to_thread(_locked, engine.unload)
    return {"status": "Memory Cleared"}

@app.get("/system/status")
def get_status():
    log_tail = ""
    try:
        with open(LOG_FILE, "r") as f:
            log_tail = "".join(f.readlines()[-40:])
    except Exception:
        pass
    return {"loaded": engine.is_loaded, "loaded_model": engine.model_path, "family": engine.family, "active_loras": engine.active_loras,
            "last_error": engine.last_error, "execution_device": str(engine.execution_device()) if engine.is_loaded else None,
            "torch_cuda_is_available": torch.cuda.is_available(), "vram": _mem_info_str(), "busy": GPU.locked(),
            "log_tail": log_tail, "progress": dict(_progress)}

@app.get("/system/progress")
def get_progress(): return dict(_progress) # Lightweight endpoint polled every 2s by the frontend during generation.

@app.get("/models")
def list_models():
    """What the transformer folder holds: Flux2 GGUF files (image) and CogVideoX-family folders (video, with each one's modes and default sizes)."""
    root = Path(TRANSFORMER_DIR)
    return {"image": [{"name": str(p.relative_to(root)), "family": "flux", "kind": "image", "modes": ["text_to_image", "image_to_image", "inpaint"]} for p in sorted(root.rglob("*.gguf"))],
            "video": video.video_models(root), "loaded": engine.model_path, "family": engine.family, "default_image_model": DEFAULT_IMAGE_MODEL}

@app.delete("/outputs/{filename:path}")
def delete_output(filename: str):
    p = _out_path(filename)
    if os.path.isdir(p): raise HTTPException(400, "a folder - delete its files, or the folder on the host")
    os.remove(p)
    return {"status": "deleted"}

# --- Generation: Flux2 Klein ---

def _generate_flux(request: GenerationRequest) -> dict:
    if not request.embed_job_id: raise HTTPException(status_code=400, detail="Provide `embed_job_id`")
    engine.ensure_image(request.model_path, request.vae_path)
    t0 = time.time()
    logger.info(f"[generate] starting. size={request.width}x{request.height} steps={request.steps} offload={request.offload_mode}")
    try: prompt_embeds = engine.load_local_embeddings(request)
    except Exception as e:
        logging.error(f"Embedding preflight failed:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Embedding validation failed: {e}")
    if request.image is None and request.image_path:
        request.image = request.image_path   # a path inside the outputs folder; _image_in reads it
        logger.info(f"Loaded base image: {request.image_path}")
    init = _image_in(request.image) if request.image else None
    if not (request.width and request.height):
        if init is None: raise HTTPException(400, "width and height are required without an init image (0 takes them from the init image)")
        request.width, request.height = _fit_size(request.width, request.height, init)
    do_cfg = request.guidance_scale > 1.0
    negative_prompt_embeds = None
    if do_cfg:
        logger.warning("CFG active on distilled model - using zeroed negative embeds.")
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    execution_seed = _seed(request.seed)
    generator = torch.Generator(device=engine.execution_device()).manual_seed(execution_seed)
    engine.set_vae_tiling(request.vae_tiling)
    engine.maybe_override_scheduler_shift(request.shift)
    engine.apply_lora_weights(request.loras)
    size = (request.width, request.height)
    pipe_kwargs = {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds, "num_inference_steps": request.steps, "guidance_scale": request.guidance_scale, "generator": generator,
                   "height": request.height, "width": request.width, "output_type": "pil", "callback_on_step_end": _make_step_callback(request.embed_job_id or "gen", request.steps), "callback_on_step_end_tensor_inputs": ["latents"]}
    if init is not None:   # img2img / inpaint: an absent mask repaints everything at the given strength
        pipe_kwargs.update({"image": init.resize(size), "mask_image": _load_mask_image(request.mask_image or "", *size), "strength": request.strength})
    else:   # txt2img: a black canvas fully repainted
        pipe_kwargs.update({"image": Image.new("RGB", size, (0, 0, 0)), "mask_image": Image.new("L", size, 255), "strength": 1.0})
    if request.reference_image: pipe_kwargs["image_reference"] = _image_in(request.reference_image)
    guards = _offload_guards(request.offload_mode, engine.pipe.transformer, engine.pipe.vae, engine.execution_device())
    print(f"--- TRACE: PIPELINE INFERENCE START (flux) ---", flush=True)
    try:
        for g in guards: g.__enter__()
        with torch.no_grad(): output = engine.pipe(**pipe_kwargs)
    finally:
        for g in reversed(guards): g.__exit__(None, None, None)
    save_dir, rel_prefix = OUTPUT_DIR, ""
    if request.sequence_dir:
        safe_dir = _UNSAFE.sub("_", request.sequence_dir)[:64]
        save_dir = os.path.join(OUTPUT_DIR, "_sequences", safe_dir)
        os.makedirs(save_dir, exist_ok=True)
        rel_prefix = f"_sequences/{safe_dir}/"
    meta_values = {"prompt": request.prompt, "width": request.width, "height": request.height, "guidance_scale": request.guidance_scale, "steps": request.steps, "seed": execution_seed, "loras": json.dumps(request.loras), "embed_job_id": request.embed_job_id, "model": engine.model_path}
    filename = _free_file(save_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{execution_seed}", ".png")
    filepath = os.path.join(save_dir, filename)
    png_info = PngInfo()
    for key in request.meta:
        if key in meta_values: png_info.add_text(key, str(meta_values[key]))
    for k in ("embed_job_id", "model"):
        if k not in request.meta and meta_values.get(k): png_info.add_text(k, str(meta_values[k]))
    output.images[0].save(filepath, format="PNG", pnginfo=png_info)
    elapsed = time.time() - t0
    logger.info(f"[generate] done in {elapsed:.1f}s -> {rel_prefix}{filename}")
    _progress.update({"status": "idle", "pct": 100})
    return {"status": "success", "message": "Pipeline complete.", "seed": execution_seed, "elapsed_seconds": round(elapsed, 2), "file_name": f"{rel_prefix}{filename}", "absolute_path": filepath, "uri_endpoint": f"/outputs/{rel_prefix}{filename}", "active_loras": engine.active_loras, "model": engine.model_path}

@app.post("/generate")
async def process_unified_generation(request: GenerationRequest): return await _gpu(_generate_flux, request)

def _sequence(request: GenerationRequest) -> dict:
    num_frames = max(2, min(int(request.model_extra.get("num_frames") or 8), 120))
    loop_gif = bool(request.model_extra.get("loop", True))
    prefix = _UNSAFE.sub("_", str(request.model_extra.get("output_prefix") or "").strip())[:40] or "seq"   # becomes a file name: only letters, digits, _ and -
    logger.info(f"[sequence] {num_frames} frames seed={request.seed}")
    frames = []
    for i in range(num_frames):
        seed = (request.seed + i) if (request.seed is not None and request.seed >= 0) else -1
        try: result = _generate_flux(request.model_copy(update={"seed": seed}))
        except video.Interrupted:
            if not frames: raise
            logger.info(f"[sequence] stopped at frame {i + 1}/{num_frames} - keeping the {len(frames)} finished")
            break
        frames.append(Image.open(os.path.join(OUTPUT_DIR, result["file_name"])).convert("RGBA"))
        logger.info(f"""[sequence] frame {i+1}/{num_frames}: {result["file_name"]}""")
        gc.collect()
        torch.cuda.empty_cache()
    out_name = _free_file(OUTPUT_DIR, f"{prefix}_{int(time.time())}", ".gif")
    frames[0].save(os.path.join(OUTPUT_DIR, out_name), format="GIF", save_all=True, append_images=frames[1:], loop=0 if loop_gif else 1, duration=120, optimize=False)
    logger.info(f"[sequence] saved {out_name}")
    return {"status": "success", "file_name": out_name, "frames": len(frames), "stopped": len(frames) < num_frames, "uri_endpoint": f"/outputs/{out_name}"}

@app.post("/generate/sequence")
async def generate_sequence_endpoint(request: GenerationRequest): return await _gpu(_sequence, request)

@app.get("/lora/debug")
def lora_debug(block: str = ""):
    if not engine.is_loaded or engine.family != "flux": return {"error": "No Flux model loaded"}
    model = engine.pipe.transformer
    module_paths = [name for name, mod in model.named_modules() if hasattr(mod, "weight") and mod.weight is not None]
    if block: return {"block_filter": block, "matched": [n for n in module_paths if block in n]}
    lora_files = {}
    try:
        for f in os.listdir(LORA_DIR):
            if f.endswith(".safetensors"):
                sd = load_file(os.path.join(LORA_DIR, f), device="cpu")
                lora_files[f] = sorted(sd.keys())[:10]
    except Exception as e:
        lora_files["error"] = str(e)
    return {"model_module_paths_sample": module_paths[:30], "total_weighted_modules": len(module_paths), "active_lora_hooks": len(engine._lora_hooks), "lora_files_sample_keys": lora_files}

# --- Generation: CogVideoX family ---

_MODE_FRAMES = {"text_to_video": (False, False), "image_to_video": (True, False), "start_end_to_video": (True, True)}   # mode -> (start frame, end frame) it takes

def _video_model(name: Optional[str], mode: str) -> dict:
    """The named model folder, or with none named the loaded video model when it can do mode, else the first folder that can."""
    models = video.video_models(Path(TRANSFORMER_DIR))
    if name:
        info = next((m for m in models if m["name"] == name), None)
        if not info: raise HTTPException(404, f"no video model folder {name!r} in {TRANSFORMER_DIR} - see /models")
        if mode not in info["modes"]: raise HTTPException(400, f"""{name} ({info["kind"]}) does {", ".join(info["modes"]) or "nothing this node can run"} - not {mode}""")
        return info
    info = next((m for m in models if m["name"] == engine.model_path and mode in m["modes"]), None) or next((m for m in models if mode in m["modes"]), None)
    if not info: raise HTTPException(404, f"no video model folder in {TRANSFORMER_DIR} can do {mode} - see /models")
    return info

def _video_size(req: VideoRequest, info: dict, start) -> tuple:
    """Unset sides take the model's defaults. A Fun InP model (multi-resolution) given a start frame keeps that frame's shape: both unset = the default pixel count at its aspect ratio, one set = the other from its aspect ratio."""
    d = info["defaults"]
    if req.width and req.height: return req.width, req.height
    if start is None or info["kind"] != "inp": return req.width or d["width"], req.height or d["height"]
    a = start.width / start.height
    if req.width: return req.width, max(16, round(req.width / a / 16) * 16)
    if req.height: return max(16, round(req.height * a / 16) * 16), req.height
    px = d["width"] * d["height"]
    return max(16, round((px * a) ** 0.5 / 16) * 16), max(16, round((px / a) ** 0.5 / 16) * 16)

def _video_plan(req: VideoRequest) -> dict:
    """Everything a video request can be refused for is checked here, before it queues for the GPU: frames vs mode, model, formats, offload, scheduler, size, input paths."""
    if req.end_image and not req.start_image: raise HTTPException(400, "an end frame needs a start frame")
    mode = req.mode or ("start_end_to_video" if req.end_image else "image_to_video" if req.start_image else "text_to_video")
    if mode not in _MODE_FRAMES: raise HTTPException(400, f"""mode {mode!r} - one of {", ".join(_MODE_FRAMES)}""")
    if _MODE_FRAMES[mode] != (bool(req.start_image), bool(req.end_image)): raise HTTPException(400, f"""{mode} takes {("a start and an end frame" if _MODE_FRAMES[mode][1] else "a start frame") if _MODE_FRAMES[mode][0] else "no frames"} - leave out what it does not take, or pick the mode that fits""")
    if mode == "text_to_video" and not req.prompt.strip(): raise HTTPException(400, "a text-only clip needs a prompt - with no frames and no words there is nothing to render")
    info = _video_model(req.model, mode)
    try: video.check_formats(req.formats)
    except ValueError as e: raise HTTPException(400, str(e))
    if req.offload_mode not in ("none", "sequential"): raise HTTPException(400, f"offload_mode {req.offload_mode!r} - one of none, sequential")
    d = info["defaults"]
    sched = req.scheduler or d["scheduler"]
    if sched not in video.SCHEDULERS: raise HTTPException(400, f"""scheduler {sched!r} - one of {", ".join(video.SCHEDULERS)}""")
    start, end = (_image_in(req.start_image) if req.start_image else None), (_image_in(req.end_image) if req.end_image else None)
    w, h, n = *_video_size(req, info, start), req.num_frames or d["num_frames"]
    if w % 16 or h % 16: raise HTTPException(400, f"width and height must be multiples of 16 (got {w}x{h})")
    if info.get("fixed_size") and [w, h] != info["fixed_size"]: raise HTTPException(400, f"""{info["name"]} has learned position embeddings and renders only at {info["fixed_size"][0]}x{info["fixed_size"][1]} (got {w}x{h})""")
    if info.get("max_frames") and video.frames_for(n, info["temporal_compression"], info["patch_size_t"]) > info["max_frames"]: raise HTTPException(400, f"""{info["name"]} has positions for at most {info["max_frames"]} frames (got {n}, which rounds to {video.frames_for(n, info["temporal_compression"], info["patch_size_t"])})""")
    return {"info": info, "mode": mode, "start": start, "end": end, "w": w, "h": h, "n": n, "fps": req.fps or d["fps"], "steps": req.steps or d["steps"], "cfg": d["guidance_scale"] if req.guidance_scale is None else req.guidance_scale, "sched": sched}

def _video(req: VideoRequest, plan: dict) -> dict:
    t0, name, w, h, steps = time.time(), plan["info"]["name"], plan["w"], plan["h"], plan["steps"]
    seed = _seed(req.seed)
    stem = _stem(req.name, seed)
    _progress.update({"step": 0, "total": steps, "pct": 0, "status": "running", "job_id": stem, "ts": time.time()})
    pe, ne = video.text_embeds(Path(TRANSFORMER_DIR) / name, req.prompt, req.negative_prompt, T5_CACHE_DIR, engine.unload)   # before the denoiser loads when possible: the encoder needs room
    engine.load(name)
    vm = engine.video
    guards = _offload_guards(req.offload_mode, vm.transformer, None, vm.device)
    logger.info(f"""[video] {name} {plan["mode"]} {w}x{h} frames={vm.frames_for(plan["n"])} steps={steps} cfg={plan["cfg"]} {plan["sched"]} seed={seed}""")
    try:
        for g in guards: g.__enter__()
        frames = vm.generate(pe, ne, plan["mode"], video.fit(plan["start"], w, h) if plan["start"] else None, video.fit(plan["end"], w, h) if plan["end"] else None, w, h, plan["n"], steps, plan["cfg"], seed, plan["sched"], req.dynamic_cfg, req.noise_aug_strength, lambda i, t: _on_step(stem, i, t))
    finally:
        for g in reversed(guards): g.__exit__(None, None, None)
    files = video.export(frames, Path(OUTPUT_DIR), stem, plan["fps"], req.formats, req.keep_frames)
    elapsed = time.time() - t0
    logger.info(f"[video] done in {elapsed:.1f}s -> {files}")
    _progress.update({"status": "idle", "pct": 100})
    return {"status": "success", "file_name": files[req.formats[0]], "files": files, "frames_dir": files.get("frames_dir", ""), "frame_count": len(frames), "fps": plan["fps"], "width": w, "height": h, "mode": plan["mode"], "model": name, "seed": seed, "elapsed_seconds": round(elapsed, 2), "uri_endpoint": f"""/outputs/{files[req.formats[0]]}"""}

@app.post("/video")
async def generate_video(request: VideoRequest): return await _gpu(_video, request, _video_plan(request))

@app.post("/video/join")
async def join_videos(request: JoinRequest):
    """Several clips into one, in order (no GPU). Each clip must have kept its frames."""
    try: video.check_formats(request.formats)
    except ValueError as e: raise HTTPException(400, str(e))
    stem = _stem(request.name or "joined", int(time.time() * 1000) % 1000)
    try: files = await asyncio.to_thread(video.join, Path(OUTPUT_DIR), request.clips, stem, request.fps, request.formats, request.drop_shared)
    except (FileNotFoundError, ValueError) as e: raise HTTPException(400, str(e))
    return {"status": "success", "file_name": files[request.formats[0]], "files": files, "frames_dir": files["frames_dir"], "frame_count": files["frame_count"], "fps": request.fps, "uri_endpoint": f"""/outputs/{files[request.formats[0]]}"""}

# --- Static outputs ---
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")   # mounted after the routes: a mount answers every method, so placed first it would take DELETE /outputs/... (405) before delete_output sees it

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

