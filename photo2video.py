#!/usr/bin/env python3
"""photo2video — transforma uma foto em vídeo localmente usando os modelos Wan2.2.

Wrapper em volta do repositório oficial Wan-Video/Wan2.2. A ideia é dar uma
interface simples (uma foto + um prompt) e cuidar de toda a chatice:

  * detectar GPU (CUDA) vs CPU;
  * escolher o modelo adequado (ti2v-5B leve, ou i2v-A14B de mais qualidade);
  * resolver/baixar o checkpoint do Hugging Face;
  * montar a linha de comando do generate.py com as flags certas de offload;
  * rodar a geração.

Esta máquina de desenvolvimento NÃO tem GPU NVIDIA. Os modelos Wan2.2 fixam o
device em `cuda:{id}` internamente (wan/textimage2video.py, wan/image2video.py),
então a geração de verdade exige uma máquina com CUDA. Aqui, sem CUDA, o wrapper
roda em modo VALIDAÇÃO/DRY-RUN: confere tudo e imprime o comando exato que você
deve rodar na máquina com GPU. Use --force para tentar mesmo assim (vai falhar/ser
inviável em CPU — é só para experimentar).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Raiz deste projeto e do repo oficial clonado ao lado deste arquivo.
ROOT = Path(__file__).resolve().parent
WAN_REPO = ROOT / "Wan2.2"
GENERATE = WAN_REPO / "generate.py"


@dataclass(frozen=True)
class ModelSpec:
    task: str            # nome da task no generate.py
    hf_repo: str         # repositório no Hugging Face
    ckpt_dirname: str    # nome da pasta local do checkpoint
    default_size: str    # tamanho padrão (área do vídeo gerado)
    t5_cpu: bool         # se deve manter o encoder T5 na CPU p/ poupar VRAM
    fps: int             # fps de amostragem do modelo
    default_frames: int  # frame_num padrão (~5s)
    approx_vram: str     # nota de VRAM para o usuário


# Apenas as tasks que fazem foto -> vídeo.
MODELS: dict[str, ModelSpec] = {
    "ti2v-5B": ModelSpec(
        task="ti2v-5B",
        hf_repo="Wan-AI/Wan2.2-TI2V-5B",
        ckpt_dirname="Wan2.2-TI2V-5B",
        default_size="1280*704",
        t5_cpu=True,
        fps=24,
        default_frames=121,
        approx_vram="~8-10GB em FP8, 24GB+ recomendado (FP16 ~27GB)",
    ),
    "i2v-A14B": ModelSpec(
        task="i2v-A14B",
        hf_repo="Wan-AI/Wan2.2-I2V-A14B",
        ckpt_dirname="Wan2.2-I2V-A14B",
        default_size="1280*720",
        t5_cpu=False,
        fps=16,
        default_frames=81,
        approx_vram="muito maior — pensado para GPUs de alta VRAM / multi-GPU",
    ),
}

DEFAULT_MODEL = "ti2v-5B"  # o mais leve; melhor default sem GPU enorme

# Faixa de duração aceita (segundos) e o ponto "confortável" dos modelos.
MIN_DURATION = 3.0
MAX_DURATION = 30.0
COMFORT_DURATION = 6.0  # acima disso os modelos saem da zona treinada (~5s)


def frames_for_duration(seconds: float, fps: int) -> int:
    """Converte segundos -> frame_num válido para o Wan.

    Os modelos Wan esperam frame_num no formato 4n+1 (ex: 81, 121). Calculamos
    frames = segundos*fps e arredondamos para o 4n+1 mais próximo (mínimo 5).
    """
    raw = seconds * fps
    n = max(1, round((raw - 1) / 4))
    return 4 * n + 1


def log(msg: str) -> None:
    print(f"[photo2video] {msg}")


def err(msg: str) -> None:
    print(f"[photo2video] ERRO: {msg}", file=sys.stderr)


def has_cuda() -> bool:
    """Detecta CUDA sem explodir caso o torch não esteja instalado."""
    try:
        import torch
    except ImportError:
        err("PyTorch não encontrado no ambiente. Ative o venv_wan.")
        return False
    return bool(torch.cuda.is_available())


def resolve_ckpt_dir(spec: ModelSpec, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    # Procura primeiro ao lado deste projeto, depois dentro do repo Wan2.2.
    for base in (ROOT, WAN_REPO):
        cand = base / spec.ckpt_dirname
        if cand.exists():
            return cand
    return ROOT / spec.ckpt_dirname  # caminho esperado se ainda não baixado


def download_checkpoint(spec: ModelSpec, ckpt_dir: Path) -> bool:
    """Baixa o checkpoint via huggingface-cli."""
    cli = shutil.which("huggingface-cli") or shutil.which("hf")
    if not cli:
        err("huggingface-cli não encontrado. Instale com: pip install 'huggingface_hub[cli]'")
        return False
    log(f"Baixando {spec.hf_repo} -> {ckpt_dir} (são vários GB, pode demorar)...")
    cmd = [cli, "download", spec.hf_repo, "--local-dir", str(ckpt_dir)]
    return subprocess.run(cmd).returncode == 0


def build_generate_cmd(args, spec: ModelSpec, ckpt_dir: Path, output: Path,
                       cuda: bool, frame_num: int) -> list[str]:
    """Monta a linha de comando para o generate.py oficial."""
    cmd = [
        sys.executable, str(GENERATE),
        "--task", spec.task,
        "--size", args.size or spec.default_size,
        "--ckpt_dir", str(ckpt_dir),
        "--image", str(Path(args.image).resolve()),
        "--prompt", args.prompt,
        "--save_file", str(output),
        "--frame_num", str(frame_num),
    ]
    if args.steps is not None:
        cmd += ["--sample_steps", str(args.steps)]
    if args.seed is not None:
        cmd += ["--base_seed", str(args.seed)]
    # convert_model_dtype => pesos em bf16 (metade da memória). Essencial em CPU também.
    cmd += ["--convert_model_dtype"]
    # offload_model move pesos entre GPU/CPU para poupar VRAM; em CPU pura não há
    # o que "descarregar" e as chamadas de sync de CUDA dariam erro, então é False.
    cmd += ["--offload_model", "True" if cuda else "False"]
    if spec.t5_cpu:
        cmd += ["--t5_cpu"]
    if args.extra:
        cmd += args.extra
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(
        prog="photo2video",
        description="Transforma uma foto em vídeo localmente com os modelos Wan2.2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  # validar/dry-run (qualquer máquina):\n"
            "  python photo2video.py -i foto.jpg -p \"a pessoa sorri e acena\" --dry-run\n\n"
            "  # baixar o checkpoint do modelo leve:\n"
            "  python photo2video.py --model ti2v-5B --download-only\n\n"
            "  # gerar de verdade (máquina com GPU NVIDIA):\n"
            "  python photo2video.py -i foto.jpg -p \"a pessoa sorri e acena\" -o saida.mp4\n"
        ),
    )
    p.add_argument("-i", "--image", help="Caminho da foto de entrada.")
    p.add_argument("-p", "--prompt", help="Descrição do movimento/cena desejada.")
    p.add_argument("-o", "--output", help="Arquivo de vídeo de saída (.mp4).")
    p.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL,
                   help=f"Modelo Wan2.2 (default: {DEFAULT_MODEL}).")
    p.add_argument("--size", help="Área do vídeo, ex: 1280*704. Default depende do modelo.")
    p.add_argument("-t", "--duration", type=float, default=5.0,
                   help=f"Duração do vídeo em segundos ({MIN_DURATION:g}-{MAX_DURATION:g}, default 5).")
    p.add_argument("--frames", type=int,
                   help="Número de frames (sobrescreve --duration; formato 4n+1).")
    p.add_argument("--steps", type=int, help="Passos de amostragem.")
    p.add_argument("--seed", type=int, help="Seed para reprodutibilidade.")
    p.add_argument("--ckpt-dir", help="Pasta do checkpoint (sobrescreve a detecção).")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Força o device. 'auto' detecta CUDA.")
    p.add_argument("--download", action="store_true",
                   help="Baixa o checkpoint se faltar, antes de gerar.")
    p.add_argument("--download-only", action="store_true",
                   help="Apenas baixa o checkpoint do modelo escolhido e sai.")
    p.add_argument("--dry-run", action="store_true",
                   help="Só valida e imprime o comando, sem executar.")
    p.add_argument("extra", nargs="*",
                   help="Flags extras repassadas cruas ao generate.py (após --).")
    args = p.parse_args()

    spec = MODELS[args.model]

    # Sanidade do ambiente.
    if not GENERATE.exists():
        err(f"generate.py não encontrado em {GENERATE}. O repo Wan2.2 foi clonado?")
        return 2

    ckpt_dir = resolve_ckpt_dir(spec, args.ckpt_dir)

    # Fluxo "só baixar".
    if args.download_only:
        log(f"Modelo: {spec.task} ({spec.hf_repo})")
        ok = download_checkpoint(spec, ckpt_dir)
        return 0 if ok else 1

    # A partir daqui, geração exige imagem e prompt.
    missing = [name for name, val in (("--image", args.image), ("--prompt", args.prompt))
               if not val]
    if missing:
        err(f"argumentos obrigatórios faltando: {', '.join(missing)}")
        return 2

    if not Path(args.image).exists():
        err(f"imagem não encontrada: {args.image}")
        return 2

    # Resolve duração -> frame_num.
    if args.frames is not None:
        frame_num = args.frames
        duration = (frame_num) / spec.fps
    else:
        if not (MIN_DURATION <= args.duration <= MAX_DURATION):
            err(f"--duration deve estar entre {MIN_DURATION:g} e {MAX_DURATION:g} segundos.")
            return 2
        frame_num = frames_for_duration(args.duration, spec.fps)
        duration = frame_num / spec.fps
    log(f"Duração: ~{duration:.1f}s  ->  {frame_num} frames @ {spec.fps}fps")
    if duration > COMFORT_DURATION:
        log(f"AVISO: {duration:.1f}s passa do ponto treinado (~5s). Espere mais custo de "
            "memória/tempo, possível queda de qualidade e deriva da semelhança da pessoa. "
            "Para vídeos longos, o ideal é encadear clipes de ~5s (ainda não implementado).")

    # Resolve device.
    if args.device == "cuda":
        cuda = True
    elif args.device == "cpu":
        cuda = False
    else:
        cuda = has_cuda()
    log(f"Device: {'CUDA (GPU)' if cuda else 'CPU'}  |  Modelo: {spec.task}")
    log(f"VRAM esperada: {spec.approx_vram}")

    # Checkpoint presente?
    if not ckpt_dir.exists():
        if args.download:
            if not download_checkpoint(spec, ckpt_dir):
                return 1
        elif args.dry_run:
            # Em dry-run não exigimos o checkpoint (são vários GB).
            log(f"AVISO: checkpoint ainda não baixado em {ckpt_dir} (ok em dry-run).")
            log(f"Para baixar: python {Path(__file__).name} --model {args.model} --download-only")
        else:
            err(f"checkpoint não encontrado em {ckpt_dir}.")
            log(f"Baixe com: python {Path(__file__).name} --model {args.model} --download-only")
            return 1

    output = Path(args.output).resolve() if args.output else (ROOT / "saida.mp4")

    cmd = build_generate_cmd(args, spec, ckpt_dir, output, cuda, frame_num)

    log("Comando a executar:")
    print("  " + " ".join(_shell_quote(c) for c in cmd))

    if not cuda:
        log("AVISO: rodando em CPU — é LENTO (pode levar de minutos a horas por clipe) "
            "e exige bastante RAM. Use poucos --steps e durações curtas para testar.")

    if args.dry_run:
        log("--dry-run: nada foi executado.")
        return 0

    log("Executando generate.py...")
    return subprocess.run(cmd, cwd=str(WAN_REPO)).returncode


def _shell_quote(s: str) -> str:
    return f'"{s}"' if (" " in s or "*" in s) else s


if __name__ == "__main__":
    raise SystemExit(main())
