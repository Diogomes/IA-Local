#!/usr/bin/env python3
"""photo2photo — edição de foto com IA mantendo a MESMA pessoa.

Edita uma foto por instrução de texto preservando a identidade da pessoa:

  * trocar a roupa (inclusive roupa de praia/biquíni, looks casuais/formais);
  * trocar o fundo / cenário / local;
  * recriar o corpo inteiro a partir de um retrato só do rosto (photo-to-photo);
  * virtual try-on: vestir a pessoa com uma peça de uma 2ª foto de referência.

Usa modelos ABERTOS e gratuitos via 🤗 diffusers, escolhidos por preservarem
bem o sujeito da foto:

  * `qwen-edit`    — Qwen/Qwen-Image-Edit-2509 (Apache-2.0, sem gate no HF).
                     Default. Aceita imagem de referência (try-on). ~20B → em
                     16GB use --quantize 4bit (ou --lowvram).
  * `flux-kontext` — black-forest-labs/FLUX.1-Kontext-dev (licença não-comercial,
                     gated no HF: aceite a licença e faça `hf auth login`). ~12B,
                     cabe bem em 16GB com offload.

Uso legítimo: edite fotos suas ou de pessoas que consentiram (moda, try-on,
restauração, recriação de cena). Esta ferramenta NÃO se destina a criar imagens
sexuais/íntimas de pessoas reais sem consentimento.

Como o photo2video, é GPU-first: sem CUDA roda em modo validação/dry-run
(modelos de 12-20B são inviáveis na CPU).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class EditModel:
    key: str
    hf_repo: str
    pipeline: str           # nome da classe de pipeline no diffusers
    uses_true_cfg: bool     # True: passa true_cfg_scale; False: guidance_scale (destilado)
    default_steps: int
    default_guidance: float
    supports_reference: bool  # aceita 2ª imagem (peça de roupa) p/ try-on
    note: str
    # componentes a quantizar em 4-bit quando --quantize 4bit
    quant_components: tuple = field(default=("transformer", "text_encoder"))


MODELS: dict[str, EditModel] = {
    "qwen-edit": EditModel(
        key="qwen-edit",
        hf_repo="Qwen/Qwen-Image-Edit-2509",
        pipeline="QwenImageEditPlusPipeline",
        uses_true_cfg=True,
        default_steps=40,
        default_guidance=4.0,
        supports_reference=True,
        note="Apache-2.0, sem gate. ~20B: em 16GB use --quantize 4bit.",
    ),
    "flux-kontext": EditModel(
        key="flux-kontext",
        hf_repo="black-forest-labs/FLUX.1-Kontext-dev",
        pipeline="FluxKontextPipeline",
        uses_true_cfg=False,
        default_steps=28,
        default_guidance=2.5,
        supports_reference=False,
        note="Não-comercial, GATED no HF (aceite a licença + hf auth login). ~12B.",
    ),
    # Usado para OUTPAINT com máscara (recriar corpo inteiro a partir do rosto).
    "flux-fill": EditModel(
        key="flux-fill",
        hf_repo="black-forest-labs/FLUX.1-Fill-dev",
        pipeline="FluxFillPipeline",
        uses_true_cfg=False,
        default_steps=28,
        default_guidance=30.0,   # Fill é guidance-distilled (valores altos ~30).
        supports_reference=False,
        note="Outpaint/inpaint com máscara. GATED no HF (aceite a licença). ~12B.",
    ),
}

DEFAULT_MODEL = "qwen-edit"

# ---------------------------------------------------------------------------
# Instruções por tarefa (em inglês — estes modelos seguem melhor; Qwen também
# entende PT/CN). A preservação da pessoa é reforçada em todas.
# ---------------------------------------------------------------------------
IDENTITY_LOCK = ("Keep the exact same person: identical face, facial features, "
                 "skin tone and identity. Photorealistic, natural and consistent.")

EDIT_NEGATIVE = ("different person, different face, identity change, face swap, "
                 "deformed face, distorted face, disfigured, bad anatomy, "
                 "extra limbs, extra fingers, fused fingers, low quality, blurry, "
                 "jpeg artifacts, watermark, text, logo, cartoon, plastic skin")

TASKS = {
    "roupa": {
        "label": "👗 Trocar roupa",
        "needs_desc": True,
        "placeholder": "ex.: biquíni de praia vermelho / terno social preto / vestido de verão floral",
        "template": ("Change only the person's clothing to: {desc}. "
                     "Keep the same hairstyle, body, pose, lighting and the same "
                     "background unchanged. Make the new outfit fit naturally. "
                     + IDENTITY_LOCK),
    },
    "fundo": {
        "label": "🏞️ Trocar fundo / cenário",
        "needs_desc": True,
        "placeholder": "ex.: praia tropical ao pôr do sol / escritório moderno / rua de Paris",
        "template": ("Replace the background / location with: {desc}. "
                     "Keep the person exactly the same — same face, same clothing, "
                     "same pose. Match the lighting to the new scene for a seamless, "
                     "realistic composite. " + IDENTITY_LOCK),
    },
    "corpo": {
        "label": "🧍 Recriar corpo inteiro (rosto → corpo)",
        "needs_desc": True,
        "placeholder": "ex.: em pé, jeans e camiseta branca, num parque / corpo inteiro de frente",
        "template": ("Generate a full-body photo of the same person, shown from "
                     "head to toe, standing, with realistic and consistent body "
                     "proportions. Scene/outfit: {desc}. The full body must be "
                     "visible in frame. " + IDENTITY_LOCK),
    },
    "tryon": {
        "label": "🧥 Vestir peça da 2ª foto (try-on)",
        "needs_desc": False,
        "placeholder": "(opcional) detalhes, ex.: ajustar o caimento, manga dobrada",
        "template": ("Dress the person from the first image with the garment shown "
                     "in the second image. Keep the same face, identity, pose and "
                     "background. Make the garment fit the body naturally. {desc} "
                     + IDENTITY_LOCK),
    },
    "livre": {
        "label": "✏️ Edição livre (descreva você)",
        "needs_desc": True,
        "placeholder": "descreva a alteração desejada…",
        "template": "{desc}. " + IDENTITY_LOCK,
    },
}

TASK_DEFAULT = "roupa"


def build_instruction(task: str, descricao: str, keep_identity: bool = True) -> str:
    """Monta a instrução final para o modelo a partir da tarefa + descrição."""
    spec = TASKS.get(task, TASKS["livre"])
    desc = (descricao or "").strip()
    if not desc and spec["needs_desc"]:
        desc = "a tasteful, photorealistic result"
    instr = spec["template"].format(desc=desc).strip()
    if not keep_identity:
        instr = instr.replace(" " + IDENTITY_LOCK, "").replace(IDENTITY_LOCK, "").strip()
    return instr


def log(msg: str) -> None:
    print(f"[photo2photo] {msg}")


def err(msg: str) -> None:
    print(f"[photo2photo] ERRO: {msg}", file=sys.stderr)


def has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        err("PyTorch não encontrado. Ative o venv (veja setup_gpu_windows.ps1).")
        return False
    return bool(torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Carregamento do pipeline (lazy: só importa diffusers/torch quando vai gerar).
# Mantemos um cache em memória para não recarregar 12-20B a cada edição.
# ---------------------------------------------------------------------------
_PIPELINE_CACHE: dict[tuple, object] = {}


def _build_quant_config(spec: EditModel):
    """4-bit (nf4) via bitsandbytes para caber em 16GB. None se indisponível."""
    try:
        import torch
        from diffusers import PipelineQuantizationConfig
        import bitsandbytes  # noqa: F401  (só para checar disponibilidade)
    except Exception:
        log("bitsandbytes/PipelineQuantizationConfig indisponível — seguindo sem "
            "quantização (use --lowvram se faltar VRAM).")
        return None
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=list(spec.quant_components),
    )


def load_editor(model: str = DEFAULT_MODEL, *, device: str | None = None,
                quantize: str | None = None, lowvram: bool = False):
    """Carrega (e cacheia) o pipeline de edição. device: 'cuda'/'cpu'/None(auto)."""
    import torch
    import diffusers

    spec = MODELS[model]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (model, device, quantize, lowvram)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    cls = getattr(diffusers, spec.pipeline)
    kwargs = {"torch_dtype": torch.bfloat16}
    if quantize == "4bit" and device == "cuda":
        qcfg = _build_quant_config(spec)
        if qcfg is not None:
            kwargs["quantization_config"] = qcfg

    log(f"Carregando {spec.hf_repo} ({spec.pipeline})… (1ª vez baixa vários GB)")
    pipe = cls.from_pretrained(spec.hf_repo, **kwargs)

    if device == "cuda":
        _place_on_cuda(pipe, lowvram=lowvram, quantized=("quantization_config" in kwargs))
        # economias de memória do VAE/atenção (quando a pipeline suporta).
        for m in ("enable_vae_tiling", "enable_vae_slicing", "enable_attention_slicing"):
            if hasattr(pipe, m):
                try:
                    getattr(pipe, m)()
                except Exception:
                    pass
    else:
        pipe = pipe.to("cpu")

    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def _place_on_cuda(pipe, *, lowvram: bool, quantized: bool) -> None:
    """Coloca o pipeline na GPU com a estratégia de memória mais robusta.

    A combinação certa de offload depende da versão do diffusers e de o modelo
    estar quantizado (bitsandbytes 4-bit) ou não. Tentamos da mais econômica de
    VRAM para a menos, caindo para a próxima se a atual não for suportada.
    """
    # Ordem de tentativas conforme o pedido do usuário.
    if lowvram:
        strategies = ["sequential", "model", "cuda"]
    elif quantized:
        # 4-bit já reduz muito a VRAM: tenta direto na GPU; offload como plano B.
        strategies = ["model", "cuda", "sequential"]
    else:
        # bf16 não-quantizado em 16GB precisa de offload.
        strategies = ["model", "sequential", "cuda"]

    last_err = None
    for strat in strategies:
        try:
            if strat == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif strat == "model":
                pipe.enable_model_cpu_offload()
            else:  # "cuda": tudo na GPU (modelos pequenos/quantizados)
                pipe.to("cuda")
            log(f"Estratégia de memória: {strat}")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"Estratégia '{strat}' indisponível ({e}); tentando a próxima…")
    # Se nada funcionou, propaga o último erro para o chamador tratar.
    if last_err is not None:
        raise last_err


def _load_image(path: str):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    # Limita o lado maior p/ não estourar memória; mantém proporção.
    max_side = 1280
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


@dataclass
class EditResult:
    path: Path
    identity_similarity: float | None = None
    identity_ok: bool = True
    attempts: int = 1
    notes: list = field(default_factory=list)


def _step_callback(progress, steps):
    if progress is None:
        return None
    def _cb(pipe_, step, t, kw):
        try:
            progress(step / max(1, int(steps)), desc="Editando")
        except Exception:
            pass
        return kw
    return _cb


def _generate_edit(pipe, spec, images_in, instruction, neg, steps, guidance,
                   seed, progress):
    """Uma passada de geração; devolve a imagem PIL."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    call = dict(image=images_in, prompt=instruction, num_inference_steps=int(steps),
                generator=gen)
    if spec.uses_true_cfg:
        call["true_cfg_scale"] = float(guidance)
        call["negative_prompt"] = neg
    else:
        call["guidance_scale"] = float(guidance)
        call["true_cfg_scale"] = 1.0
    cb = _step_callback(progress, steps)
    if cb is not None:
        call["callback_on_step_end"] = cb
    return pipe(**call).images[0]


def _finalize(pil, *, src_image, output, upscale, face_restore, device, notes):
    """Aplica melhoria de qualidade (opcional) e salva; devolve o Path."""
    if face_restore or (upscale and upscale > 1):
        try:
            import enhance
            dev = "cuda" if (device in (None, "cuda")) else device
            if face_restore:
                pil = enhance.restore_faces_pil(pil, device=dev)
            if upscale and upscale > 1:
                pil = enhance.upscale_pil(pil, scale=int(upscale), device=dev)
            notes.append(f"qualidade: rosto={'on' if face_restore else 'off'}, "
                         f"upscale={upscale}x")
        except Exception as e:  # noqa: BLE001
            notes.append(f"melhoria de qualidade pulada ({e})")
    OUTPUTS.mkdir(exist_ok=True)
    out = Path(output).resolve() if output else (OUTPUTS / f"edit_{uuid.uuid4().hex[:8]}.png")
    pil.save(out)
    return out


def edit_photo(*, image: str, instruction: str, model: str = DEFAULT_MODEL,
               reference: str | None = None, steps: int | None = None,
               guidance: float | None = None, seed: int = 42,
               negative: str | None = None, output: str | None = None,
               device: str | None = None, quantize: str | None = None,
               lowvram: bool = False, progress=None,
               upscale: int = 1, face_restore: bool = False,
               identity_check: bool = False,
               identity_threshold: float = 0.45,
               max_retries: int = 1) -> EditResult:
    """Edita a foto preservando a pessoa. Devolve um EditResult.

    Se identity_check estiver ligado, mede a similaridade facial entrada×saída e,
    se ficar abaixo do limiar, tenta de novo com outra seed (até max_retries),
    guardando o melhor resultado. upscale/face_restore melhoram a qualidade final.
    """
    spec = MODELS[model]
    steps = steps or spec.default_steps
    guidance = spec.default_guidance if guidance is None else guidance
    neg = EDIT_NEGATIVE + (", " + negative.strip() if (negative or "").strip() else "")

    pipe = load_editor(model, device=device, quantize=quantize, lowvram=lowvram)

    img = _load_image(image)
    images_in = img
    if reference and spec.supports_reference:
        images_in = [img, _load_image(reference)]
    elif reference and not spec.supports_reference:
        log(f"AVISO: {model} não aceita imagem de referência; ignorando-a. "
            "Use --model qwen-edit para try-on com peça de referência.")

    import identity as idmod
    do_check = identity_check and idmod.available()
    attempts = (max_retries + 1) if do_check else 1

    best = None  # (sim_or_-1, pil, sim, ok, attempt_idx)
    notes: list = []
    OUTPUTS.mkdir(exist_ok=True)
    for attempt in range(attempts):
        s = int(seed) + attempt
        pil = _generate_edit(pipe, spec, images_in, instruction, neg, steps,
                             guidance, s, progress)
        sim, ok = (None, True)
        if do_check:
            tmp = OUTPUTS / f".cand_{uuid.uuid4().hex[:6]}.png"
            pil.save(tmp)
            sim, ok = idmod.check(image, str(tmp), identity_threshold)
            try:
                tmp.unlink()
            except OSError:
                pass
            rank = sim if sim is not None else -1.0
            if best is None or rank > best[0]:
                best = (rank, pil, sim, ok, attempt + 1)
            if sim is None or ok:
                break
            log(f"Identidade baixa ({sim:.3f} < {identity_threshold}); "
                f"tentativa {attempt + 2}/{attempts}…")
        else:
            best = (-1.0, pil, None, True, attempt + 1)
            break

    _, pil, sim, ok, used = best
    if do_check and sim is not None:
        notes.append(f"identidade={sim:.3f} ({'ok' if ok else 'baixa'}), tentativas={used}")
        if not ok:
            notes.append("⚠️ a pessoa pode ter mudado — tente outra seed/guidance menor.")

    out = _finalize(pil, src_image=image, output=output, upscale=upscale,
                    face_restore=face_restore, device=device, notes=notes)
    return EditResult(path=out, identity_similarity=sim, identity_ok=ok,
                      attempts=used, notes=notes)


def _build_outpaint_canvas(img, *, top_ratio: float = 0.34, side_ratio: float = 0.18):
    """Cria a tela estendida + máscara p/ recriar o corpo a partir do rosto.

    Posiciona a foto original no topo-centro de uma tela mais alta; a máscara é
    BRANCA (área a gerar) em tudo, exceto sobre a foto original (preto = manter).
    """
    from PIL import Image
    w, h = img.size
    # múltiplos de 16 ajudam o FLUX; dimensiona a tela.
    new_w = int(round(w * (1 + 2 * side_ratio) / 16) * 16)
    new_h = int(round(h / top_ratio / 16) * 16)
    ox = (new_w - w) // 2
    oy = int(new_h * 0.04)  # pequena margem no topo
    canvas = Image.new("RGB", (new_w, new_h), (127, 127, 127))
    canvas.paste(img, (ox, oy))
    mask = Image.new("L", (new_w, new_h), 255)        # tudo a gerar…
    keep = Image.new("L", (w, h), 0)
    mask.paste(keep, (ox, oy))                        # …menos a foto original
    return canvas, mask


def outpaint_full_body(*, image: str, describe: str = "", model: str = "flux-fill",
                       steps: int | None = None, guidance: float | None = None,
                       seed: int = 42, output: str | None = None,
                       device: str | None = None, quantize: str | None = None,
                       lowvram: bool = False, progress=None,
                       upscale: int = 1, face_restore: bool = False,
                       identity_check: bool = False,
                       identity_threshold: float = 0.45) -> EditResult:
    """Recria o corpo inteiro a partir de um retrato usando outpaint com máscara."""
    import torch
    spec = MODELS[model]
    if spec.pipeline != "FluxFillPipeline":
        raise ValueError("outpaint_full_body requer um modelo de fill (flux-fill).")
    steps = steps or spec.default_steps
    guidance = spec.default_guidance if guidance is None else guidance

    pipe = load_editor(model, device=device, quantize=quantize, lowvram=lowvram)
    img = _load_image(image)
    canvas, mask = _build_outpaint_canvas(img)

    prompt = (f"A full-body photo of the same person from head to toe, standing, "
              f"realistic and consistent body and proportions. {describe.strip()}. "
              + IDENTITY_LOCK)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    call = dict(image=canvas, mask_image=mask, prompt=prompt,
                height=canvas.height, width=canvas.width,
                num_inference_steps=int(steps), guidance_scale=float(guidance),
                generator=gen)
    cb = _step_callback(progress, steps)
    if cb is not None:
        call["callback_on_step_end"] = cb
    pil = pipe(**call).images[0]

    notes: list = ["modo: outpaint com máscara (FLUX Fill)"]
    sim, ok = (None, True)
    import identity as idmod
    if identity_check and idmod.available():
        OUTPUTS.mkdir(exist_ok=True)
        tmp = OUTPUTS / f".cand_{uuid.uuid4().hex[:6]}.png"
        pil.save(tmp)
        sim, ok = idmod.check(image, str(tmp), identity_threshold)
        try:
            tmp.unlink()
        except OSError:
            pass
        if sim is not None:
            notes.append(f"identidade={sim:.3f} ({'ok' if ok else 'baixa'})")

    out = _finalize(pil, src_image=image, output=output, upscale=upscale,
                    face_restore=face_restore, device=device, notes=notes)
    return EditResult(path=out, identity_similarity=sim, identity_ok=ok, notes=notes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        prog="photo2photo",
        description="Edita uma foto mantendo a mesma pessoa (roupa/fundo/corpo/try-on).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  # trocar a roupa:\n"
            "  python photo2photo.py -i foto.jpg --task roupa "
            "-d \"biquíni de praia azul\" -o saida.png\n\n"
            "  # rosto -> corpo inteiro:\n"
            "  python photo2photo.py -i rosto.jpg --task corpo "
            "-d \"em pé, jeans e camiseta branca\"\n\n"
            "  # try-on com peça de referência (só qwen-edit):\n"
            "  python photo2photo.py -i pessoa.jpg --ref camisa.jpg --task tryon\n\n"
            "  # validar sem rodar (qualquer máquina):\n"
            "  python photo2photo.py -i foto.jpg --task roupa -d \"...\" --dry-run\n"
        ),
    )
    p.add_argument("-i", "--image", help="Foto de entrada (a pessoa a preservar).")
    p.add_argument("--task", choices=list(TASKS), default=TASK_DEFAULT,
                   help=f"Tipo de edição (default: {TASK_DEFAULT}).")
    p.add_argument("-d", "--describe", default="",
                   help="Descrição da mudança (roupa/fundo/cena).")
    p.add_argument("--instruction",
                   help="Instrução crua (ignora --task/--describe se informado).")
    p.add_argument("--ref", "--reference", dest="reference",
                   help="2ª foto: peça de roupa de referência (try-on; só qwen-edit).")
    p.add_argument("-o", "--output", help="PNG de saída (default em outputs/).")
    p.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL,
                   help=f"Modelo de edição (default: {DEFAULT_MODEL}).")
    p.add_argument("--steps", type=int, help="Passos de difusão.")
    p.add_argument("--guidance", type=float, help="Aderência à instrução.")
    p.add_argument("--seed", type=int, default=42, help="Seed.")
    p.add_argument("-n", "--negative", help="Termos extras a evitar.")
    p.add_argument("--no-keep-identity", action="store_true",
                   help="Não reforçar a preservação da pessoa.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--quantize", choices=["none", "4bit"], default="4bit",
                   help="4bit (nf4) faz o modelo caber em 16GB (default).")
    p.add_argument("--lowvram", action="store_true",
                   help="Offload sequencial (menos VRAM, mais lento).")
    # Qualidade (pós-processamento)
    p.add_argument("--enhance", action="store_true",
                   help="Melhorar a saída: restaurar rosto + upscale.")
    p.add_argument("--scale", type=int, choices=[1, 2, 4], default=2,
                   help="Fator de upscale quando --enhance (default 2).")
    p.add_argument("--no-face-restore", action="store_true",
                   help="Com --enhance, NÃO restaurar rosto (só upscale).")
    # Fidelidade medida
    p.add_argument("--check-identity", action="store_true",
                   help="Medir similaridade facial e retentar se a pessoa mudar.")
    p.add_argument("--identity-threshold", type=float, default=0.45,
                   help="Limiar de similaridade (default 0.45).")
    p.add_argument("--retries", type=int, default=2,
                   help="Máx. de retentativas com --check-identity (default 2).")
    # Corpo inteiro de verdade
    p.add_argument("--outpaint", action="store_true",
                   help="Tarefa 'corpo': usar outpaint com máscara (FLUX Fill).")
    p.add_argument("--download-only", action="store_true",
                   help="Só baixa os pesos do modelo escolhido e sai.")
    p.add_argument("--dry-run", action="store_true",
                   help="Valida e imprime a instrução, sem carregar o modelo.")
    args = p.parse_args()

    spec = MODELS[args.model]

    if args.download_only:
        log(f"Baixando pesos de {spec.hf_repo} …")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(spec.hf_repo)
        except Exception as e:
            err(f"falha ao baixar: {e}")
            return 1
        log("Concluído.")
        return 0

    if not args.image:
        err("--image é obrigatório.")
        return 2
    if not Path(args.image).exists():
        err(f"imagem não encontrada: {args.image}")
        return 2

    keep_identity = not args.no_keep_identity
    instruction = args.instruction or build_instruction(
        args.task, args.describe, keep_identity=keep_identity)

    device = (None if args.device == "auto"
              else args.device)
    cuda = has_cuda() if args.device == "auto" else (args.device == "cuda")
    quantize = None if args.quantize == "none" else args.quantize

    log(f"Modelo: {args.model} ({spec.hf_repo})")
    log(f"Tarefa: {args.task}  |  Device: {'CUDA' if cuda else 'CPU'}")
    log(f"Instrução: {instruction}")
    if args.reference:
        log(f"Referência (try-on): {args.reference}")

    if not cuda:
        log("AVISO: edição com modelos de 12-20B é INVIÁVEL na CPU (use a máquina "
            "com a RTX 5070 Ti). Em GPU, --quantize 4bit faz caber em 16GB.")

    if args.dry_run:
        log("--dry-run: nada foi executado.")
        return 0

    if not cuda:
        err("sem CUDA — abortando (rode na máquina com GPU). Use --dry-run para validar.")
        return 1

    face_restore = args.enhance and not args.no_face_restore
    scale = args.scale if args.enhance else 1
    common = dict(
        image=args.image, model=args.model, steps=args.steps, guidance=args.guidance,
        seed=args.seed, output=args.output, device=device, quantize=quantize,
        lowvram=args.lowvram, upscale=scale, face_restore=face_restore,
        identity_check=args.check_identity, identity_threshold=args.identity_threshold)
    try:
        if args.task == "corpo" and args.outpaint:
            model_fill = args.model if MODELS[args.model].pipeline == "FluxFillPipeline" else "flux-fill"
            common["model"] = model_fill
            log(f"Outpaint com máscara usando {model_fill}.")
            res = outpaint_full_body(describe=args.describe, **common)
        else:
            res = edit_photo(instruction=instruction, reference=args.reference,
                             negative=args.negative, max_retries=args.retries, **common)
    except Exception as e:
        err(f"falha na edição: {e}")
        return 1
    for n in res.notes:
        log(n)
    log(f"✅ Pronto: {res.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
