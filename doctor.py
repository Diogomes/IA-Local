#!/usr/bin/env python3
"""doctor — diagnóstico do ambiente da ferramenta (rode no PC da GPU).

Confere, em PT, tudo que precisa estar no lugar para gerar com qualidade:
PyTorch + CUDA (incl. Blackwell/RTX 50 = sm_120), VRAM, libs de edição e os
recursos opcionais (upscale/rosto/identidade), além do repo Wan2.2 e do
checkpoint. Com `--smoke`, roda uma geração MÍNIMA de vídeo de verdade (só se
houver GPU e checkpoint) para confirmar o pipeline ponta a ponta.

Uso:
    venv_wan\\Scripts\\python doctor.py            (Windows)
    venv_wan/bin/python doctor.py                  (Linux)
    ... adicione --smoke para o teste de geração mínimo.
"""

from __future__ import annotations

import importlib.util as iutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}

# RTX 5070 Ti = Blackwell = sm_120. Builds cu121/cu124 não têm kernels p/ ela.
BLACKWELL = (12, 0)


def _has(mod: str) -> bool:
    try:
        return iutil.find_spec(mod) is not None
    except Exception:
        return False


def _ver(pkg: str) -> str:
    try:
        import importlib.metadata as m
        return m.version(pkg)
    except Exception:
        return "?"


def check() -> list:
    """Devolve uma lista de (status, título, detalhe)."""
    rows: list = []

    # --- Python ---
    vi = sys.version_info
    pyok = (vi.major, vi.minor) == (3, 12)
    rows.append((OK if pyok else WARN, f"Python {vi.major}.{vi.minor}",
                 "ok (3.12)" if pyok else "recomendado 3.12 (PyTorch não tem wheels p/ 3.13/3.14)"))

    # --- PyTorch + CUDA ---
    if not _has("torch"):
        rows.append((FAIL, "PyTorch", "não instalado — rode setup_gpu_windows.ps1"))
        return rows
    import torch
    rows.append((OK, "PyTorch", f"{torch.__version__} (CUDA build: {torch.version.cuda})"))

    if not torch.cuda.is_available():
        rows.append((FAIL, "CUDA", "indisponível — confira o driver NVIDIA e o torch cu128"))
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        archs = []
        try:
            archs = torch.cuda.get_arch_list()
        except Exception:
            pass
        rows.append((OK, "GPU", f"{name}  (sm_{cap[0]}{cap[1]})"))
        try:
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            rows.append((OK if vram >= 12 else WARN, "VRAM",
                         f"{vram:.1f} GB" + ("" if vram >= 12 else " (pouco; use 540p e --quantize 4bit)")))
        except Exception:
            pass
        # Blackwell precisa de sm_120 no build.
        if cap >= BLACKWELL:
            tag = f"sm_{cap[0]}{cap[1]}"
            has_kernel = any(tag in a for a in archs) or any("sm_120" in a or "sm_90" in a for a in archs)
            rows.append((OK if has_kernel else FAIL, "Suporte Blackwell",
                         (f"build cobre {tag}" if has_kernel else
                          f"o build do torch NÃO cobre {tag} — reinstale com "
                          "--index-url https://download.pytorch.org/whl/cu128")))

    # --- Atenção (flash-attn opcional; SDPA é o fallback) ---
    rows.append((OK, "flash-attn",
                 "instalado" if _has("flash_attn") else "ausente (ok — usa SDPA, ótimo em CUDA)"))

    # --- Libs de difusão/edição ---
    for pkg, lo in (("diffusers", "0.35"), ("transformers", ""), ("accelerate", "")):
        if _has(pkg.replace("-", "_")):
            rows.append((OK, pkg, _ver(pkg)))
        else:
            rows.append((FAIL, pkg, "ausente — pip install -r requirements_cuda.txt"))
    # pipelines de edição existem nesta versão do diffusers?
    if _has("diffusers"):
        import diffusers
        miss = [c for c in ("QwenImageEditPlusPipeline", "FluxKontextPipeline",
                            "FluxFillPipeline") if not hasattr(diffusers, c)]
        rows.append((OK if not miss else WARN, "Pipelines de edição",
                     "todas presentes" if not miss else f"faltam: {', '.join(miss)} (atualize o diffusers)"))

    # --- Quantização 4-bit (p/ caber o editor em 16GB) ---
    rows.append((OK if _has("bitsandbytes") else WARN, "bitsandbytes (4-bit)",
                 _ver("bitsandbytes") if _has("bitsandbytes") else
                 "ausente — sem 4-bit o editor pode não caber em 16GB (pip install bitsandbytes)"))

    # --- Recursos opcionais de qualidade/fidelidade ---
    rows.append((OK if _has("spandrel") else WARN, "Upscale (spandrel)",
                 "ok" if _has("spandrel") else "ausente — usa Lanczos (requirements_enhance.txt)"))
    rows.append((OK if _has("gfpgan") else WARN, "Restaurar rosto (gfpgan)",
                 "ok" if _has("gfpgan") else "ausente — sem restauração de rosto (requirements_enhance.txt)"))
    idok = _has("insightface") and _has("onnxruntime")
    rows.append((OK if idok else WARN, "Checar identidade (insightface)",
                 "ok" if idok else "ausente — checagem de identidade desativada (requirements_enhance.txt)"))

    # --- Mídia ---
    for pkg, label in (("cv2", "OpenCV"), ("PIL", "Pillow"), ("imageio", "imageio")):
        rows.append((OK if _has(pkg) else WARN, label,
                     "ok" if _has(pkg) else "ausente"))

    # --- Wan2.2 + checkpoint ---
    gen = ROOT / "Wan2.2" / "generate.py"
    rows.append((OK if gen.exists() else FAIL, "Repo Wan2.2",
                 "ok" if gen.exists() else "generate.py não encontrado"))
    ckpt = ROOT / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth"
    rows.append((OK if ckpt.exists() else WARN, "Checkpoint ti2v-5B",
                 "baixado" if ckpt.exists() else
                 "ausente — python photo2video.py --model ti2v-5B --download-only"))

    return rows


def _smoke_video() -> tuple:
    """Geração mínima de vídeo (256px, 13 frames, 4 passos). Só com GPU+checkpoint."""
    import photo2video as p2v
    import subprocess
    img = ROOT / "Wan2.2" / "examples" / "i2v_input.JPG"
    if not img.exists():
        return WARN, "Smoke vídeo", "imagem de exemplo não encontrada; pulado"
    out = ROOT / "outputs" / "smoke.mp4"
    out.parent.mkdir(exist_ok=True)
    cmd, ck = p2v.generation_command(
        image=str(img), prompt="a pessoa sorri levemente", size="256*256",
        frame_num=13, steps=4, output=out, cuda=True)
    if not ck.exists():
        return WARN, "Smoke vídeo", "checkpoint ausente; pulado"
    r = subprocess.run(cmd, cwd=str(p2v.WAN_REPO))
    okv = out.exists() and out.stat().st_size > 0
    return (OK if okv else FAIL, "Smoke vídeo",
            f"gerou {out.name}" if okv else f"falhou (código {r.returncode})")


def main() -> int:
    smoke = "--smoke" in sys.argv
    print("=== doctor — diagnóstico da ferramenta ===\n")
    rows = check()
    worst_ok = True
    for st, title, detail in rows:
        if st == FAIL:
            worst_ok = False
        print(f"  {_MARK[st]} {title:28s} {detail}")

    # Smoke test (opcional) — só faz sentido com GPU.
    if smoke:
        print("\n--- smoke test (geração mínima) ---")
        try:
            import torch
            if torch.cuda.is_available():
                st, title, detail = _smoke_video()
                print(f"  {_MARK[st]} {title:28s} {detail}")
            else:
                print("  ⚠️  Smoke vídeo               pulado (sem CUDA)")
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ Smoke vídeo               erro: {e}")

    print("\n" + ("✅ Pronto para gerar." if worst_ok else
                  "❌ Há itens FAIL acima — resolva-os antes de gerar."))
    return 0 if worst_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
