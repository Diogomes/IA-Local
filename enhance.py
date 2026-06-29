#!/usr/bin/env python3
"""enhance — pós-processamento de QUALIDADE para foto e vídeo.

Dois passos, ambos opcionais e independentes:

  * upscale (super-resolução) com Real-ESRGAN via `spandrel` — aumenta a
    resolução preservando o conteúdo (bom para identidade);
  * restauração de rosto com GFPGAN — recupera detalhes faciais.

Tudo é carregado de forma PREGUIÇOSA e com degradação graciosa: se uma lib não
estiver instalada (ou quebrar no Windows), a função cai num fallback e segue a
vida — o app nunca quebra por causa disto. Veja requirements_enhance.txt.

As libs de restauração de rosto (gfpgan/basicsr/facexlib) importam um módulo
`torchvision.transforms.functional_tensor` que foi REMOVIDO no torchvision novo;
aplicamos um shim de compatibilidade antes de importá-las.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Pesos (baixados sob demanda via huggingface_hub / URL oficial).
REALESRGAN_REPO = "ai-forever/Real-ESRGAN"
REALESRGAN_FILES = {2: "RealESRGAN_x2.pth", 4: "RealESRGAN_x4.pth", 8: "RealESRGAN_x8.pth"}
GFPGAN_URL = ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/"
              "GFPGANv1.4.pth")

_UPSCALE_CACHE: dict = {}
_GFPGAN_CACHE: dict = {}


def log(msg: str) -> None:
    print(f"[enhance] {msg}")


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _functional_tensor_shim() -> None:
    """Cria torchvision.transforms.functional_tensor se faltar (gfpgan/basicsr)."""
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import importlib.util as u
        if u.find_spec(name) is not None:
            return
    except Exception:
        pass
    try:
        import types
        import torchvision.transforms.functional as F
        shim = types.ModuleType(name)
        # basicsr só usa rgb_to_grayscale daqui.
        shim.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules[name] = shim
    except Exception:
        pass


def backends() -> dict:
    """Diz quais recursos estão disponíveis no ambiente atual."""
    import importlib.util as u
    return {
        "upscale": u.find_spec("spandrel") is not None,
        "face": u.find_spec("gfpgan") is not None,
    }


# ---------------------------------------------------------------------------
# Upscale (Real-ESRGAN via spandrel)
# ---------------------------------------------------------------------------
def _load_upscaler(scale: int, device: str):
    key = (scale, device)
    if key in _UPSCALE_CACHE:
        return _UPSCALE_CACHE[key]
    from huggingface_hub import hf_hub_download
    from spandrel import ModelLoader
    import torch

    fname = REALESRGAN_FILES.get(scale, REALESRGAN_FILES[4])
    path = hf_hub_download(REALESRGAN_REPO, fname)
    model = ModelLoader().load_from_file(path)
    model.to(torch.device(device)).eval()
    _UPSCALE_CACHE[key] = model
    return model


def upscale_pil(img, scale: int = 2, device: str | None = None):
    """Aumenta a resolução da imagem PIL. Fallback p/ Lanczos se Real-ESRGAN faltar."""
    from PIL import Image
    device = device or _device()
    try:
        import torch
        import numpy as np
        model = _load_upscaler(scale, device)
        arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(ten)
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((out * 255.0 + 0.5).astype("uint8"))
    except Exception as e:  # noqa: BLE001
        log(f"upscale Real-ESRGAN indisponível ({e}); usando Lanczos.")
        w, h = img.size
        return img.resize((w * scale, h * scale), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Restauração de rosto (GFPGAN)
# ---------------------------------------------------------------------------
def _load_gfpgan(device: str):
    if "r" in _GFPGAN_CACHE:
        return _GFPGAN_CACHE["r"]
    _functional_tensor_shim()
    from gfpgan import GFPGANer
    restorer = GFPGANer(model_path=GFPGAN_URL, upscale=1, arch="clean",
                        channel_multiplier=2, bg_upsampler=None, device=device)
    _GFPGAN_CACHE["r"] = restorer
    return restorer


def restore_faces_pil(img, weight: float = 0.5, device: str | None = None):
    """Restaura rostos numa imagem PIL. Devolve a imagem original se GFPGAN faltar."""
    from PIL import Image
    device = device or _device()
    try:
        import numpy as np
        import cv2
        restorer = _load_gfpgan(device)
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        _, _, out = restorer.enhance(bgr, has_aligned=False, only_center_face=False,
                                     paste_back=True, weight=weight)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as e:  # noqa: BLE001
        log(f"restauração de rosto indisponível ({e}); mantendo a imagem.")
        return img


# ---------------------------------------------------------------------------
# Orquestração: imagem e vídeo
# ---------------------------------------------------------------------------
def enhance_image(in_path: str, out_path: str | None = None, *, scale: int = 2,
                  face_restore: bool = True, face_weight: float = 0.5,
                  device: str | None = None) -> Path:
    from PIL import Image
    device = device or _device()
    img = Image.open(in_path).convert("RGB")
    if face_restore:
        img = restore_faces_pil(img, weight=face_weight, device=device)
    if scale and scale > 1:
        img = upscale_pil(img, scale=scale, device=device)
    out = Path(out_path).resolve() if out_path else Path(in_path).with_name(
        Path(in_path).stem + f"_hq.png")
    img.save(out)
    return out


def enhance_video(in_path: str, out_path: str | None = None, *, scale: int = 2,
                  face_restore: bool = True, face_weight: float = 0.5,
                  device: str | None = None, progress=None) -> Path:
    """Aplica upscale/restauração quadro a quadro e reencoda o vídeo."""
    import imageio
    from PIL import Image
    device = device or _device()

    reader = imageio.get_reader(in_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 24)
    out = Path(out_path).resolve() if out_path else Path(in_path).with_name(
        Path(in_path).stem + "_hq.mp4")
    writer = imageio.get_writer(out, fps=fps, codec="libx264",
                                quality=8, macro_block_size=None)
    try:
        n = reader.count_frames()
    except Exception:
        n = 0
    try:
        for i, frame in enumerate(reader):
            img = Image.fromarray(frame).convert("RGB")
            if face_restore:
                img = restore_faces_pil(img, weight=face_weight, device=device)
            if scale and scale > 1:
                img = upscale_pil(img, scale=scale, device=device)
            import numpy as np
            writer.append_data(np.asarray(img))
            if progress is not None and n:
                try:
                    progress((i + 1) / n, desc="Melhorando vídeo")
                except Exception:
                    pass
    finally:
        writer.close()
        reader.close()
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Melhora qualidade de foto/vídeo (upscale + rosto).")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output")
    p.add_argument("--scale", type=int, default=2, choices=[1, 2, 4])
    p.add_argument("--no-face", action="store_true", help="Não restaurar rosto.")
    p.add_argument("--video", action="store_true", help="Tratar entrada como vídeo.")
    a = p.parse_args()
    log(f"Backends: {backends()}")
    fn = enhance_video if a.video else enhance_image
    out = fn(a.input, a.output, scale=a.scale, face_restore=not a.no_face)
    log(f"✅ {out}")
