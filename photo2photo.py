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
        # offload p/ caber em 16GB: sequential é o mais econômico (e mais lento).
        if lowvram:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        for m in ("enable_vae_tiling", "enable_vae_slicing", "enable_attention_slicing"):
            if hasattr(pipe, m):
                getattr(pipe, m)()
    else:
        pipe = pipe.to("cpu")

    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


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


def edit_photo(*, image: str, instruction: str, model: str = DEFAULT_MODEL,
               reference: str | None = None, steps: int | None = None,
               guidance: float | None = None, seed: int = 42,
               negative: str | None = None, output: str | None = None,
               device: str | None = None, quantize: str | None = None,
               lowvram: bool = False, progress=None) -> Path:
    """Edita a foto e devolve o caminho do PNG gerado."""
    import torch

    spec = MODELS[model]
    steps = steps or spec.default_steps
    guidance = spec.default_guidance if guidance is None else guidance
    neg = (negative or "").strip()
    neg = (EDIT_NEGATIVE + (", " + neg if neg else ""))

    pipe = load_editor(model, device=device, quantize=quantize, lowvram=lowvram)

    img = _load_image(image)
    images_in = img
    if reference and spec.supports_reference:
        images_in = [img, _load_image(reference)]
    elif reference and not spec.supports_reference:
        log(f"AVISO: {model} não aceita imagem de referência; ignorando-a. "
            "Use --model qwen-edit para try-on com peça de referência.")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    call = dict(image=images_in, prompt=instruction, num_inference_steps=int(steps),
                generator=gen)
    if spec.uses_true_cfg:
        call["true_cfg_scale"] = float(guidance)
        call["negative_prompt"] = neg
    else:
        call["guidance_scale"] = float(guidance)
        # Flux Kontext: negativo só atua com true_cfg_scale>1.
        call["true_cfg_scale"] = 1.0
    if progress is not None:
        def _cb(pipe_, step, t, kw):
            try:
                progress(step / max(1, int(steps)), desc="Editando")
            except Exception:
                pass
            return kw
        call["callback_on_step_end"] = _cb

    out_img = pipe(**call).images[0]

    OUTPUTS.mkdir(exist_ok=True)
    out = Path(output).resolve() if output else (OUTPUTS / f"edit_{uuid.uuid4().hex[:8]}.png")
    out_img.save(out)
    return out


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

    try:
        out = edit_photo(
            image=args.image, instruction=instruction, model=args.model,
            reference=args.reference, steps=args.steps, guidance=args.guidance,
            seed=args.seed, negative=args.negative, output=args.output,
            device=device, quantize=quantize, lowvram=args.lowvram)
    except Exception as e:
        err(f"falha na edição: {e}")
        return 1
    log(f"✅ Pronto: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
