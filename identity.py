#!/usr/bin/env python3
"""identity — mede se a pessoa foi PRESERVADA entre duas fotos.

Usa embeddings faciais do InsightFace (modelo buffalo_l, baixado sob demanda) e
calcula a similaridade de cosseno entre o rosto da foto de entrada e o da saída.
Serve para avisar/retentar quando uma edição "trocou" a pessoa.

Tudo é preguiçoso e degrada com elegância: sem insightface/onnxruntime, as
funções devolvem None (similaridade desconhecida) e o app segue normalmente.
Veja requirements_enhance.txt.
"""

from __future__ import annotations

# Acima disto consideramos "provavelmente a mesma pessoa" (cosseno de embeddings
# L2-normalizados do buffalo_l). ~0.45-0.5 é um limiar usual.
DEFAULT_THRESHOLD = 0.45

_APP_CACHE: dict = {}


def log(msg: str) -> None:
    print(f"[identity] {msg}")


def available() -> bool:
    import importlib.util as u
    return (u.find_spec("insightface") is not None
            and u.find_spec("onnxruntime") is not None)


def _get_app():
    if "app" in _APP_CACHE:
        return _APP_CACHE["app"]
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    # ctx_id=0 tenta GPU (onnxruntime-gpu); cai para CPU (-1) se não houver.
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        app.prepare(ctx_id=-1, det_size=(640, 640))
    _APP_CACHE["app"] = app
    return app


def _embedding(path: str):
    import cv2
    app = _get_app()
    img = cv2.imread(path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    # maior rosto (área da bbox) = sujeito principal.
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return faces[0].normed_embedding


def face_similarity(path_a: str, path_b: str) -> float | None:
    """Cosseno entre os rostos principais de duas imagens (0..1). None se indisponível."""
    if not available():
        return None
    try:
        import numpy as np
        ea, eb = _embedding(path_a), _embedding(path_b)
        if ea is None or eb is None:
            log("rosto não detectado em uma das imagens; pulando checagem.")
            return None
        return float(np.dot(ea, eb))  # embeddings já normalizados
    except Exception as e:  # noqa: BLE001
        log(f"checagem indisponível ({e}).")
        return None


def check(input_path: str, output_path: str, threshold: float = DEFAULT_THRESHOLD):
    """Retorna (similaridade|None, ok: bool). ok=True quando >= threshold ou desconhecida."""
    sim = face_similarity(input_path, output_path)
    if sim is None:
        return None, True
    return sim, (sim >= threshold)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Mede preservação de identidade entre duas fotos.")
    p.add_argument("-a", "--input", required=True, help="Foto original.")
    p.add_argument("-b", "--output", required=True, help="Foto editada.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    a = p.parse_args()
    log(f"InsightFace disponível: {available()}")
    sim, ok = check(a.input, a.output, a.threshold)
    if sim is None:
        log("Similaridade: desconhecida (lib ausente ou sem rosto).")
    else:
        log(f"Similaridade: {sim:.3f}  ->  {'MESMA pessoa (ok)' if ok else 'MUDOU (abaixo do limiar)'}")
