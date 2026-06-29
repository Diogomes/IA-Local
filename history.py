#!/usr/bin/env python3
"""history — histórico persistente das gerações (lê a pasta outputs/).

Tudo que a ferramenta gera (edições, etapas do Estúdio, lotes, vídeos, versões
HQ) é salvo em `outputs/`. Este módulo apenas LÊ essa pasta e devolve listas
ordenadas da mais recente para a mais antiga — então o histórico é persistente
entre sessões sem precisar de banco de dados. Arquivos temporários (começando
com ".") são ignorados.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VID_EXT = {".mp4", ".webm", ".mov", ".gif"}


def _scan(exts: set, limit: int) -> list:
    if not OUTPUTS.exists():
        return []
    files = [p for p in OUTPUTS.iterdir()
             if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def images(limit: int = 60) -> list:
    """Caminhos das imagens geradas, mais recentes primeiro."""
    return [str(p) for p in _scan(IMG_EXT, limit)]


def videos(limit: int = 40) -> list:
    """Caminhos dos vídeos gerados, mais recentes primeiro."""
    return [str(p) for p in _scan(VID_EXT, limit)]


if __name__ == "__main__":
    print(f"outputs/: {OUTPUTS}")
    imgs, vids = images(), videos()
    print(f"{len(imgs)} imagem(ns), {len(vids)} vídeo(s).")
    for p in imgs[:10]:
        print("  🖼️ ", p)
    for p in vids[:10]:
        print("  🎬 ", p)
