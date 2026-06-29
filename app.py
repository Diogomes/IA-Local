#!/usr/bin/env python3
"""Interface web (Gradio) do photo2video — foto + prompt -> vídeo com Wan2.2.

Envie uma foto, escreva um prompt e gere um vídeo curto e realista. A fidelidade
à pessoa vem do I2V do Wan (a foto vira o quadro inicial) + um reforço de
identidade nos prompts (positivo e negativo) que esta UI adiciona por padrão.

QUALIDADE: o ti2v-5B foi treinado em 720p com ~50 passos. Numa GPU NVIDIA
(ex.: RTX 5070 Ti 16GB) use 540p/720p e 40-50 passos para qualidade alta.
Resoluções baixas (256/384/512) servem só para TESTAR o pipeline na CPU.

Cada geração chama o generate.py oficial como UM subprocesso, então a memória
(T5 + DiT) é toda liberada ao terminar.

Rodar:  venv_wan/bin/python app.py          (Linux/macOS)
        venv_wan\\Scripts\\python app.py     (Windows)
        -> abre em http://127.0.0.1:7860
"""

from __future__ import annotations

import re
import subprocess
import uuid
import os
from pathlib import Path

import gradio as gr

from photo2video import FPS_DEFAULT, WAN_REPO, generation_command, has_cuda
import photo2photo as p2p
import history

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "gigaverse3d-logo.png"

CUDA = has_cuda()

# Rótulo amigável -> valor de --size.
#  - 720p/540p/480p = qualidade real (GPU).
#  - 256/384/512    = só teste do pipeline (CPU).
RESOLUCOES = {
    "720p — máxima qualidade (1280×704, GPU)": "1280*704",
    "540p — alta qualidade, mais leve (960×544, GPU)": "960*544",
    "480p — boa qualidade, mais rápido (832×480, GPU)": "832*480",
    "512px — teste (qualidade fraca)": "512*512",
    "384px — teste (fraco)": "384*384",
    "256px — teste rápido (CPU)": "256*256",
}

# preset -> (resolução, passos, duração s, guidance)
PRESETS_GPU = {
    "Máxima qualidade (GPU)": ("720p — máxima qualidade (1280×704, GPU)", 50, 5.0, 5.0),
    "Alta qualidade, mais rápida (GPU)": ("540p — alta qualidade, mais leve (960×544, GPU)", 40, 5.0, 5.0),
    "Equilíbrio (GPU)": ("480p — boa qualidade, mais rápido (832×480, GPU)", 30, 4.0, 5.0),
}
PRESETS_CPU = {
    "Teste rápido (CPU)": ("256px — teste rápido (CPU)", 8, 2.0, 5.0),
    "Teste melhor (CPU)": ("512px — teste (qualidade fraca)", 20, 3.0, 5.0),
}
PRESETS = {**PRESETS_GPU, **PRESETS_CPU} if CUDA else {**PRESETS_CPU, **PRESETS_GPU}
PRESET_DEFAULT = "Máxima qualidade (GPU)" if CUDA else "Teste rápido (CPU)"

PROMPT_EXEMPLO = ("a pessoa sorri suavemente e acena para a câmera, luz natural, "
                  "movimento sutil e realista, alta qualidade")


def frames_para_duracao(segundos: float) -> int:
    raw = segundos * FPS_DEFAULT
    n = max(1, round((raw - 1) / 4))
    return 4 * n + 1


def aplicar_preset(preset: str):
    if preset not in PRESETS:
        return gr.update(), gr.update(), gr.update(), gr.update()
    res, passos, dur, guide = PRESETS[preset]
    return (gr.update(value=res), gr.update(value=passos),
            gr.update(value=dur), gr.update(value=guide))


def _stream_video(imagem_path, prompt, size, frames, passos, guidance,
                  manter_pessoa, negativo_extra, melhorar_video, escala_video,
                  progress):
    """Gera o vídeo (subprocesso Wan) e faz streaming do progresso.

    Generator que produz (video_path_ou_None, status). Reaproveitado pela aba de
    vídeo e pelo Estúdio (animar no fim).
    """
    w, h = (int(x) for x in size.split("*"))
    shift = 3.0 if w * h <= 512 * 512 else 5.0
    saida = OUTPUTS / f"video_{uuid.uuid4().hex[:8]}.mp4"

    cmd, _ = generation_command(
        image=imagem_path, prompt=prompt.strip(), size=size, frame_num=frames,
        steps=int(passos), output=saida, guide_scale=float(guidance),
        shift=shift, cuda=CUDA, keep_identity=bool(manter_pessoa),
        negative_prompt=(negativo_extra or "").strip() or None)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(WAN_REPO))

    fase_difusao = False
    ultima_erro = ""
    passo_re = re.compile(r"(\d+)/(\d+)\s*\[")
    for linha in proc.stdout:
        linha = linha.rstrip()
        if "out of memory" in linha.lower() or "OutOfMemory" in linha:
            ultima_erro = linha
        if "Creating WanModel" in linha or "loading" in linha.lower():
            yield None, "📦 Carregando o modelo na memória…"
        elif "Generating video" in linha:
            fase_difusao = True
            yield None, "🎬 Gerando (difusão)… isso é o mais demorado."
        else:
            m = passo_re.search(linha)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                if fase_difusao:
                    progress(cur / tot, desc="Difusão")
                    yield None, f"🎬 Difusão: passo {cur}/{tot}"
                else:
                    yield None, f"📦 Carregando modelo: parte {cur}/{tot}"

    proc.wait()
    if saida.exists() and saida.stat().st_size > 0:
        if melhorar_video:
            yield str(saida), ("✅ Vídeo gerado. ✨ Melhorando qualidade "
                               f"(restaurar rosto + upscale {int(escala_video)}×)…")
            try:
                import enhance
                hq = enhance.enhance_video(str(saida), scale=int(escala_video),
                                           face_restore=True, device="cuda",
                                           progress=progress)
                yield str(hq), (f"✅ Pronto (HQ {int(escala_video)}×)! "
                                f"~{frames/FPS_DEFAULT:.1f}s, salvo em {hq.name}.")
                return
            except Exception as e:  # noqa: BLE001
                yield str(saida), (f"✅ Vídeo pronto, mas a melhoria falhou ({e}). "
                                   "Instale requirements_enhance.txt no PC da GPU.")
                return
        yield str(saida), f"✅ Pronto! Vídeo de ~{frames/FPS_DEFAULT:.1f}s ({size})."
    elif ultima_erro:
        yield None, ("❌ Falta de memória de GPU (OOM). Tente: resolução menor "
                     "(540p ou 480p), menos frames (duração menor) ou menos passos. "
                     "Em 16GB, 720p com 5s pode estourar — 540p costuma caber folgado.\n"
                     f"Detalhe: {ultima_erro}")
    else:
        yield None, ("❌ A geração falhou (código %s). Veja o terminal para o log. "
                     "Causas comuns: checkpoint não baixado, PyTorch sem suporte à "
                     "GPU (Blackwell exige cu128) ou falta de memória." % proc.returncode)


def gerar(imagem_path, prompt, resolucao_label, passos, duracao, guidance,
          manter_pessoa, negativo_extra, melhorar_video, escala_video,
          progress=gr.Progress()):
    if not imagem_path:
        yield None, "⚠️ Envie uma foto primeiro."
        return
    if not prompt or not prompt.strip():
        yield None, "⚠️ Escreva um prompt descrevendo o movimento/cena."
        return

    size = RESOLUCOES.get(resolucao_label, "256*256")
    frames = frames_para_duracao(float(duracao))
    yield None, (f"⏳ Iniciando… {size}, {frames} frames (~{frames/FPS_DEFAULT:.1f}s), "
                 f"{int(passos)} passos, guidance {guidance}.\n"
                 + ("🎯 Preservação da pessoa: LIGADA.\n" if manter_pessoa else "")
                 + ("" if CUDA else "Em CPU é LENTO: carregar o modelo já leva ~1 min "
                    "e cada passo demora; resoluções altas podem levar horas."))
    yield from _stream_video(imagem_path, prompt, size, frames, passos, guidance,
                             manter_pessoa, negativo_extra, melhorar_video,
                             escala_video, progress)


# --- Edição de foto (roupa / fundo / corpo / try-on) -----------------------
EDIT_TASK_LABELS = {v["label"]: k for k, v in p2p.TASKS.items()}
EDIT_MODEL_LABELS = {
    "Qwen-Image-Edit (livre, sem gate) — recomendado": "qwen-edit",
    "FLUX.1 Kontext (gated no HF, não-comercial)": "flux-kontext",
}


def _galeria_de_variantes(variants):
    """Lista (caminho, legenda com a similaridade) p/ a galeria, melhor primeiro."""
    itens = []
    for i, v in enumerate(variants):
        if v.identity_similarity is not None:
            cap = f"#{i + 1} • id {v.identity_similarity:.2f}"
        else:
            cap = f"#{i + 1}"
        itens.append((str(v.path), cap))
    return itens


def editar(imagem_path, ref_path, task_label, descricao, manter_pessoa,
           modelo_label, passos, guidance, seed, negativo, lowvram,
           melhorar, escala, checar_id, outpaint, n_var, progress=gr.Progress()):
    if not imagem_path:
        yield None, None, "⚠️ Envie uma foto primeiro."
        return
    if not CUDA:
        yield None, None, ("⚠️ A edição usa modelos de 12–20B e só roda na GPU "
                           "(RTX 5070 Ti). Nesta máquina (CPU) é inviável. Rode o "
                           "app no PC com a GPU.")
        return

    task = EDIT_TASK_LABELS.get(task_label, p2p.TASK_DEFAULT)
    model = EDIT_MODEL_LABELS.get(modelo_label, p2p.DEFAULT_MODEL)
    instruction = p2p.build_instruction(task, descricao, keep_identity=bool(manter_pessoa))
    ref = ref_path if (task == "tryon" or ref_path) else None
    usar_outpaint = bool(outpaint) and task == "corpo"
    n_var = max(1, int(n_var))
    quality = dict(upscale=int(escala) if melhorar else 1,
                   face_restore=bool(melhorar), identity_check=bool(checar_id))

    yield None, None, (f"⏳ Carregando o modelo (a 1ª vez baixa vários GB e demora). "
                       "Edição em 4-bit p/ caber em 16GB…\n"
                       + ("🧍 Outpaint com máscara (FLUX Fill).\n" if usar_outpaint else "")
                       + (f"🎲 Gerando {n_var} variações.\n" if (n_var > 1 and not usar_outpaint) else "")
                       + f"Instrução: {instruction}")
    try:
        if usar_outpaint:
            res = p2p.outpaint_full_body(
                image=imagem_path, describe=descricao, seed=int(seed),
                steps=int(passos) or None, device="cuda", quantize="4bit",
                lowvram=bool(lowvram), progress=progress, **quality)
        elif n_var > 1:
            variants = p2p.generate_variations(
                image=imagem_path, instruction=instruction, model=model, n=n_var,
                reference=ref, steps=int(passos) or None, guidance=float(guidance),
                seed=int(seed), negative=(negativo or "").strip() or None,
                device="cuda", quantize="4bit", lowvram=bool(lowvram),
                progress=progress, **quality)
            best = variants[0]
            extra = ("\n" + "\n".join(best.notes)) if best.notes else ""
            yield (str(best.path), _galeria_de_variantes(variants),
                   f"✅ {len(variants)} variações (melhor 1ª, por identidade). "
                   f"Melhor: {best.path.name}.{extra}")
            return
        else:
            res = p2p.edit_photo(
                image=imagem_path, instruction=instruction, model=model, reference=ref,
                steps=int(passos) or None, guidance=float(guidance), seed=int(seed),
                negative=(negativo or "").strip() or None, device="cuda",
                quantize="4bit", lowvram=bool(lowvram), progress=progress, **quality)
    except Exception as e:
        yield None, None, (f"❌ Falhou: {e}\nDicas: marque 'Low-VRAM' se for OOM; para "
                           "FLUX Fill/Kontext aceite a licença no HF e rode `hf auth login`.")
        return

    extra = ("\n" + "\n".join(res.notes)) if res.notes else ""
    aviso = ("\n⚠️ A identidade pode ter mudado — tente outra seed."
             if (res.identity_similarity is not None and not res.identity_ok) else "")
    yield str(res.path), [str(res.path)], f"✅ Pronto! Salvo em {res.path.name}.{extra}{aviso}"


def estudio(imagem_path, fullbody, fb_desc, outpaint, roupa, fundo, keep,
            modelo_label, passos, guidance, seed, lowvram, melhorar, escala,
            checkid, animar, video_prompt, progress=gr.Progress()):
    if not imagem_path:
        yield None, None, None, "⚠️ Envie uma foto primeiro."
        return
    if not CUDA:
        yield None, None, None, ("⚠️ O Estúdio usa modelos de 12–20B e só roda na "
                                 "GPU (RTX 5070 Ti). Rode o app no PC com a GPU.")
        return
    if not (fullbody or (roupa or "").strip() or (fundo or "").strip() or melhorar):
        yield None, None, None, ("⚠️ Escolha ao menos uma transformação (corpo / "
                                 "roupa / fundo) ou marque melhorar qualidade.")
        return

    model = EDIT_MODEL_LABELS.get(modelo_label, p2p.DEFAULT_MODEL)
    etapas = " → ".join([s for s in [
        "corpo" if fullbody else "", "roupa" if (roupa or "").strip() else "",
        "fundo" if (fundo or "").strip() else "",
        "qualidade" if melhorar else ""] if s]) or "qualidade"
    if animar:
        etapas += " → 🎬 animar"
    yield None, None, None, (f"⏳ Pipeline: {etapas}. Cada etapa usa o modelo (a 1ª "
                             "vez baixa vários GB). Isso pode levar alguns minutos…")
    try:
        res = p2p.studio_transform(
            image=imagem_path, model=model, steps=int(passos) or None,
            guidance=float(guidance), seed=int(seed), device="cuda", quantize="4bit",
            lowvram=bool(lowvram), progress=progress, keep_identity=bool(keep),
            full_body=bool(fullbody), full_body_desc=fb_desc, outpaint=bool(outpaint),
            roupa=roupa, fundo=fundo, upscale=int(escala) if melhorar else 1,
            face_restore=bool(melhorar), identity_check=bool(checkid))
    except Exception as e:
        yield None, None, None, (f"❌ Falhou: {e}\nDicas: marque 'Low-VRAM' se for "
                                 "OOM; para FLUX Fill aceite a licença no HF e rode "
                                 "`hf auth login`.")
        return
    extra = "\n".join(res.notes)
    aviso = ("\n⚠️ A identidade pode ter mudado — tente outra seed."
             if (res.identity_similarity is not None and not res.identity_ok) else "")
    base_status = f"✅ Imagem pronta: {res.path.name}.\n{extra}{aviso}"
    yield str(res.path), res.steps, None, base_status

    if not animar:
        return
    vp = (video_prompt or "").strip() or PROMPT_EXEMPLO
    # Anima o resultado final em 540p ~5s (cabe bem na 5070 Ti).
    size = RESOLUCOES["540p — alta qualidade, mais leve (960×544, GPU)"]
    frames = frames_para_duracao(5.0)
    for vid, vstatus in _stream_video(
            str(res.path), vp, size, frames, 40, 5.0, True, None, False, 2, progress):
        yield str(res.path), res.steps, vid, f"{base_status}\n— {vstatus}"


LOTE_TASK_LABELS = {
    "👗 Trocar roupa": "roupa",
    "🏞️ Trocar fundo / cenário": "fundo",
    "✏️ Livre (cada linha = uma instrução)": "livre",
}


def lote(imagem_path, task_label, itens_texto, keep, modelo_label, passos, guidance,
         seed, lowvram, melhorar, escala, checkid, progress=gr.Progress()):
    if not imagem_path:
        yield None, "⚠️ Envie uma foto primeiro."
        return
    if not CUDA:
        yield None, ("⚠️ O lote usa modelos de 12–20B e só roda na GPU "
                     "(RTX 5070 Ti). Rode o app no PC com a GPU.")
        return
    itens = [ln.strip() for ln in (itens_texto or "").splitlines() if ln.strip()]
    if not itens:
        yield None, "⚠️ Escreva ao menos um item (um por linha)."
        return

    task = LOTE_TASK_LABELS.get(task_label, "roupa")
    model = EDIT_MODEL_LABELS.get(modelo_label, p2p.DEFAULT_MODEL)
    yield None, (f"⏳ Gerando {len(itens)} resultados de '{task}' (a 1ª vez baixa o "
                 "modelo). Mesma seed em todos — só o texto muda.")
    try:
        res = p2p.batch_edit(
            image=imagem_path, task=task, items=itens, model=model,
            steps=int(passos) or None, guidance=float(guidance), seed=int(seed),
            keep_identity=bool(keep), device="cuda", quantize="4bit",
            lowvram=bool(lowvram), progress=progress,
            upscale=int(escala) if melhorar else 1, face_restore=bool(melhorar),
            identity_check=bool(checkid))
    except Exception as e:
        yield None, (f"❌ Falhou: {e}\nDicas: marque 'Low-VRAM' se for OOM.")
        return

    galeria = []
    for desc, r in res:
        cap = desc + (f" • id {r.identity_similarity:.2f}"
                      if r.identity_similarity is not None else "")
        galeria.append((str(r.path), cap))
    yield galeria, f"✅ {len(galeria)} resultados. Clique para ampliar; salvos em outputs/."


_GPU_TXT = ("**Dispositivo: GPU (CUDA) ✅** — pode usar 540p/720p e 40-50 passos."
            if CUDA else
            "**Dispositivo: CPU (lento) ⚠️** — use os presets de *teste*. "
            "Qualidade real (540p/720p) só numa GPU NVIDIA.")

AVISO_QUALIDADE = (
    "> 🎯 **Fidelidade à pessoa:** a foto vira o **quadro inicial** e esta UI ainda "
    "adiciona um reforço de identidade nos prompts. Mantenha **\"Preservar a pessoa\"** "
    "ligado e descreva no prompt **só o movimento/cena**, não as feições.\n"
    ">\n"
    "> 🎬 **Qualidade x velocidade:** o modelo foi treinado em **720p, ~50 passos**. "
    "Para vídeo nítido e realista use **540p/720p e 40-50 passos** (numa GPU sai em "
    "minutos). Resoluções 256/384/512 servem só para *testar* o pipeline.")


APP_CSS = """
:root {
    --gigaverse-panel: rgba(8, 14, 24, 0.84);
    --gigaverse-line: rgba(61, 161, 255, 0.36);
    --gigaverse-blue: #1689ff;
    --gigaverse-cyan: #53d8ff;
    --gigaverse-silver: #d8dde8;
    --gigaverse-muted: #8d99ab;
}

.gradio-container {
    min-height: 100vh;
    color: var(--gigaverse-silver) !important;
    background:
        radial-gradient(circle at 50% 0%, rgba(22, 137, 255, 0.18), transparent 32rem),
        linear-gradient(135deg, #03050a 0%, #08111d 45%, #020409 100%) !important;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(83, 216, 255, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(83, 216, 255, 0.04) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: linear-gradient(to bottom, black 0%, transparent 80%);
}

#gigaverse-shell {
    max-width: 1240px;
    margin: 0 auto;
}

.hero {
    align-items: center;
    gap: 28px;
    padding: 24px 26px;
    margin: 8px 0 18px;
    border: 1px solid rgba(83, 216, 255, 0.28);
    border-radius: 8px;
    background:
        linear-gradient(145deg, rgba(10, 18, 31, 0.96), rgba(3, 7, 14, 0.92)),
        linear-gradient(90deg, rgba(22, 137, 255, 0.12), transparent);
    box-shadow: 0 0 34px rgba(22, 137, 255, 0.18), inset 0 0 28px rgba(83, 216, 255, 0.06);
}

.logo-mark img {
    object-fit: contain !important;
    filter: drop-shadow(0 0 18px rgba(22, 137, 255, 0.7));
}

img.logo-mark {
    width: min(100%, 225px);
    height: auto;
    display: block;
    border-radius: 6px;
    filter: drop-shadow(0 0 18px rgba(22, 137, 255, 0.7));
}

.brand-copy h1 {
    margin: 0;
    color: #f1f5fb;
    font-size: clamp(2rem, 4vw, 4.6rem);
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: 0;
    text-transform: uppercase;
    text-shadow: 0 0 22px rgba(22, 137, 255, 0.52);
}

.brand-copy p {
    margin: 12px 0 0;
    color: var(--gigaverse-muted);
    font-size: 1rem;
}

.brand-copy .tagline {
    margin-top: 14px;
    color: var(--gigaverse-cyan);
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}

.notice {
    margin-bottom: 18px;
    border-left: 3px solid var(--gigaverse-blue);
    padding: 10px 16px;
    color: #b9c6d8;
    background: rgba(9, 17, 30, 0.68);
}

.work-panel {
    padding: 18px;
    border: 1px solid var(--gigaverse-line);
    border-radius: 8px;
    background: var(--gigaverse-panel);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 42px rgba(0,0,0,0.34);
}

.work-panel label,
.work-panel span,
.work-panel p {
    color: var(--gigaverse-silver) !important;
}

textarea,
input,
.work-panel .block,
.work-panel .input-container,
.work-panel .gradio-dropdown,
.work-panel .gradio-radio,
.work-panel .gradio-video,
.work-panel .gradio-image {
    border-color: rgba(83, 216, 255, 0.22) !important;
    background: rgba(2, 6, 12, 0.7) !important;
}

button.primary,
.work-panel button.primary {
    border: 1px solid rgba(83, 216, 255, 0.75) !important;
    background: linear-gradient(180deg, #1aa2ff 0%, #075bdc 100%) !important;
    color: white !important;
    box-shadow: 0 0 22px rgba(22, 137, 255, 0.42);
    font-weight: 800 !important;
}

button.primary:hover {
    filter: brightness(1.12);
}

.gradio-container footer {
    display: none !important;
}

@media (max-width: 760px) {
    .hero {
        padding: 18px;
    }

    .brand-copy h1 {
        font-size: 2.2rem;
    }
}
"""


with gr.Blocks(title="Gigaverse3d photo to video", elem_id="gigaverse-shell") as demo:
    with gr.Row(elem_classes=["hero"]):
        with gr.Column(scale=1, min_width=180):
            gr.HTML(
                f"<img class='logo-mark' src='/gradio_api/file={LOGO_PATH.as_posix()}' "
                "alt='Gigaverse3d logo'>"
            )
        with gr.Column(scale=4, min_width=320):
            gr.HTML(
                "<div class='brand-copy'>"
                "<h1>Gigaverse3d</h1>"
                "<div class='tagline'>Impressao 3D • Tecnologia • Universo</div>"
                "<p>Photo to video com preservacao visual da imagem de entrada. "
                f"Dispositivo detectado: <strong>{'GPU (CUDA)' if CUDA else 'CPU (lento)'}</strong></p>"
                "</div>"
            )
    with gr.Tabs() as tabs:
        # ---------------- Aba 1: Foto -> Vídeo ----------------
        with gr.Tab("🎬 Foto → Vídeo", id="video"):
            gr.Markdown(AVISO_QUALIDADE, elem_classes=["notice"])
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    imagem = gr.Image(type="filepath", label="Foto de entrada", height=300)
                    prompt = gr.Textbox(label="Prompt (o que deve acontecer no vídeo)", lines=3,
                                        placeholder=PROMPT_EXEMPLO, value=PROMPT_EXEMPLO)
                    manter_pessoa = gr.Checkbox(
                        value=True, label="🎯 Preservar a pessoa (rosto/identidade) — recomendado")
                    preset = gr.Radio(choices=list(PRESETS) + ["Personalizado"],
                                      value=PRESET_DEFAULT, label="Preset")

                    with gr.Accordion("Ajustes avançados", open=False):
                        resolucao = gr.Dropdown(choices=list(RESOLUCOES),
                                                value=PRESETS[PRESET_DEFAULT][0],
                                                label="Resolução")
                        passos = gr.Slider(4, 60, value=PRESETS[PRESET_DEFAULT][1], step=1,
                                           label="Passos de difusão (↑ = melhor e mais lento)")
                        duracao = gr.Slider(1.0, 10.0, value=PRESETS[PRESET_DEFAULT][2], step=0.5,
                                            label="Duração (segundos) — ideal ≤5s")
                        guidance = gr.Slider(1.0, 12.0, value=PRESETS[PRESET_DEFAULT][3], step=0.5,
                                             label="Guidance (aderência ao prompt; ~5 é o ideal)")
                        negativo_extra = gr.Textbox(
                            label="Prompt negativo extra (o que evitar — opcional)", lines=2,
                            placeholder="ex.: fundo cheio de gente, texto na tela, cores saturadas")
                        melhorar_video = gr.Checkbox(
                            value=False,
                            label="✨ Melhorar qualidade do vídeo (restaurar rosto + upscale) — mais lento")
                        escala_video = gr.Radio(choices=[1, 2, 4], value=2,
                                                label="Upscale (×) do vídeo")

                    botao = gr.Button("🎬 Gerar vídeo", variant="primary")

                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    video = gr.Video(label="Resultado", height=360)
                    status = gr.Textbox(label="Status", interactive=False, lines=6)

            preset.change(aplicar_preset, inputs=preset,
                          outputs=[resolucao, passos, duracao, guidance], api_name=False)
            botao.click(gerar,
                        inputs=[imagem, prompt, resolucao, passos, duracao, guidance,
                                manter_pessoa, negativo_extra, melhorar_video, escala_video],
                        outputs=[video, status], api_name="gerar")

        # ---------------- Aba 2: Editar foto ----------------
        with gr.Tab("🖼️ Editar foto (roupa / fundo / corpo)", id="editar"):
            gr.Markdown(
                "> 🎯 Edita a foto **mantendo a mesma pessoa**: trocar roupa "
                "(inclui roupa de praia), trocar o fundo/cenário, ou recriar o "
                "**corpo inteiro** a partir de um retrato do rosto.\n"
                "> Use fotos suas ou de quem consentiu. **Precisa de GPU** "
                "(roda na RTX 5070 Ti em 4-bit).", elem_classes=["notice"])
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    ed_imagem = gr.Image(type="filepath", label="Foto de entrada (a pessoa)", height=280)
                    ed_task = gr.Radio(choices=list(EDIT_TASK_LABELS),
                                       value=list(EDIT_TASK_LABELS)[0], label="O que fazer")
                    ed_desc = gr.Textbox(label="Descrição da mudança", lines=2,
                                         placeholder=p2p.TASKS[p2p.TASK_DEFAULT]["placeholder"])
                    ed_ref = gr.Image(type="filepath", visible=False, height=160,
                                      label="Peça de roupa de referência (try-on)")
                    ed_keep = gr.Checkbox(value=True,
                                          label="🎯 Preservar a pessoa — recomendado")

                    with gr.Accordion("Ajustes avançados", open=False):
                        ed_model = gr.Dropdown(choices=list(EDIT_MODEL_LABELS),
                                               value=list(EDIT_MODEL_LABELS)[0], label="Modelo")
                        ed_steps = gr.Slider(10, 60, value=p2p.MODELS[p2p.DEFAULT_MODEL].default_steps,
                                             step=1, label="Passos")
                        ed_guidance = gr.Slider(1.0, 10.0,
                                                value=p2p.MODELS[p2p.DEFAULT_MODEL].default_guidance,
                                                step=0.5, label="Aderência à instrução")
                        ed_seed = gr.Number(value=42, label="Seed", precision=0)
                        ed_neg = gr.Textbox(label="Evitar (negativo extra — opcional)", lines=2)
                        ed_lowvram = gr.Checkbox(value=False,
                                                 label="Low-VRAM (mais lento; use se der OOM)")
                        ed_melhorar = gr.Checkbox(value=True,
                                                  label="✨ Melhorar qualidade (restaurar rosto + upscale)")
                        ed_escala = gr.Radio(choices=[1, 2, 4], value=2,
                                             label="Upscale (×) quando melhorar")
                        ed_checkid = gr.Checkbox(value=True,
                                                 label="🔎 Checar identidade (retenta se a pessoa mudar)")
                        ed_outpaint = gr.Checkbox(value=False,
                                                  label="🧍 Corpo: outpaint com máscara (FLUX Fill, mais consistente)")
                        ed_var = gr.Slider(1, 6, value=1, step=1,
                                           label="🎲 Variações (gera N e ordena pela fidelidade)")

                    ed_botao = gr.Button("🖼️ Editar foto", variant="primary")

                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    ed_saida = gr.Image(type="filepath", label="Resultado (melhor)", height=320)
                    ed_gallery = gr.Gallery(label="Variações (melhor → pior)", columns=3, height=160)
                    ed_status = gr.Textbox(label="Status", interactive=False, lines=6)
                    with gr.Row():
                        ed_to_video = gr.Button("🎬 Animar este resultado")
                        ed_reuse = gr.Button("♻️ Usar como nova entrada")

            def _toggle_ref(task_label):
                is_tryon = EDIT_TASK_LABELS.get(task_label) == "tryon"
                ph = p2p.TASKS.get(EDIT_TASK_LABELS.get(task_label, "livre"), {}).get("placeholder", "")
                return gr.update(visible=is_tryon), gr.update(placeholder=ph)

            ed_task.change(_toggle_ref, inputs=ed_task, outputs=[ed_ref, ed_desc], api_name=False)
            ed_botao.click(editar,
                           inputs=[ed_imagem, ed_ref, ed_task, ed_desc, ed_keep,
                                   ed_model, ed_steps, ed_guidance, ed_seed, ed_neg, ed_lowvram,
                                   ed_melhorar, ed_escala, ed_checkid, ed_outpaint, ed_var],
                           outputs=[ed_saida, ed_gallery, ed_status], api_name="editar")

            # Clicar numa variação da galeria a promove para o "Resultado (melhor)".
            def _escolher_variante(evt: gr.SelectData):
                try:
                    return evt.value.get("image", {}).get("path") or evt.value
                except Exception:
                    return gr.update()
            ed_gallery.select(_escolher_variante, inputs=None, outputs=ed_saida, api_name=False)

        # ---------------- Aba 3: Estúdio (1 clique) ----------------
        with gr.Tab("✨ Estúdio (transformação completa)", id="estudio"):
            gr.Markdown(
                "> ✨ **Um clique** encadeia tudo, preservando a pessoa: recriar "
                "**corpo inteiro** → trocar **roupa** → trocar **fundo** → "
                "**melhorar qualidade**. Preencha só o que quiser mudar.\n"
                "> **Precisa de GPU.** Cada etapa usa o modelo de edição.",
                elem_classes=["notice"])
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    st_imagem = gr.Image(type="filepath", label="Foto de entrada", height=260)
                    st_fullbody = gr.Checkbox(value=False,
                                              label="🧍 Recriar corpo inteiro (a partir do rosto)")
                    st_fb_desc = gr.Textbox(label="Corpo: cena/pose (opcional)", lines=1,
                                            placeholder="ex.: em pé, de frente, num parque")
                    st_outpaint = gr.Checkbox(value=False,
                                              label="↳ usar outpaint com máscara (FLUX Fill)")
                    st_roupa = gr.Textbox(label="👗 Nova roupa (deixe vazio p/ não trocar)", lines=1,
                                          placeholder="ex.: biquíni de praia / terno social")
                    st_fundo = gr.Textbox(label="🏞️ Novo fundo (deixe vazio p/ não trocar)", lines=1,
                                          placeholder="ex.: praia tropical ao pôr do sol")
                    st_keep = gr.Checkbox(value=True, label="🎯 Preservar a pessoa — recomendado")
                    st_animar = gr.Checkbox(value=False,
                                            label="🎬 Animar no fim (gera um vídeo 540p ~5s do resultado)")
                    st_vprompt = gr.Textbox(label="Prompt do vídeo (se animar)", lines=2,
                                            placeholder=PROMPT_EXEMPLO)

                    with gr.Accordion("Ajustes avançados", open=False):
                        st_model = gr.Dropdown(choices=list(EDIT_MODEL_LABELS),
                                               value=list(EDIT_MODEL_LABELS)[0], label="Modelo (roupa/fundo)")
                        st_steps = gr.Slider(10, 60, value=p2p.MODELS[p2p.DEFAULT_MODEL].default_steps,
                                             step=1, label="Passos por etapa")
                        st_guidance = gr.Slider(1.0, 10.0,
                                                value=p2p.MODELS[p2p.DEFAULT_MODEL].default_guidance,
                                                step=0.5, label="Aderência")
                        st_seed = gr.Number(value=42, label="Seed", precision=0)
                        st_lowvram = gr.Checkbox(value=False, label="Low-VRAM")
                        st_melhorar = gr.Checkbox(value=True,
                                                  label="✨ Melhorar qualidade no fim (rosto + upscale)")
                        st_escala = gr.Radio(choices=[1, 2, 4], value=2, label="Upscale (×)")
                        st_checkid = gr.Checkbox(value=True, label="🔎 Checar identidade no fim")

                    st_botao = gr.Button("✨ Transformar", variant="primary")

                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    st_saida = gr.Image(type="filepath", label="Resultado final", height=300)
                    st_video = gr.Video(label="Vídeo (se animar)", height=240)
                    st_gallery = gr.Gallery(label="Etapas (galeria)", columns=4, height=140)
                    st_status = gr.Textbox(label="Status", interactive=False, lines=5)
                    st_to_video = gr.Button("🎬 Animar este resultado (na aba de vídeo)")

            st_botao.click(estudio,
                           inputs=[st_imagem, st_fullbody, st_fb_desc, st_outpaint,
                                   st_roupa, st_fundo, st_keep, st_model, st_steps,
                                   st_guidance, st_seed, st_lowvram, st_melhorar,
                                   st_escala, st_checkid, st_animar, st_vprompt],
                           outputs=[st_saida, st_gallery, st_video, st_status],
                           api_name="estudio")

        # ---------------- Aba 4: Lote (vários looks/cenários) ----------------
        with gr.Tab("🗂️ Lote (vários looks/cenários)", id="lote"):
            gr.Markdown(
                "> 🗂️ Aplica a **mesma** tarefa a **vários textos de uma vez** "
                "(um por linha) na mesma foto — ótimo para comparar looks ou "
                "cenários. Mesma pessoa, mesma seed; só o texto muda.\n"
                "> **Precisa de GPU.**", elem_classes=["notice"])
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    lo_imagem = gr.Image(type="filepath", label="Foto de entrada", height=260)
                    lo_task = gr.Radio(choices=list(LOTE_TASK_LABELS),
                                       value=list(LOTE_TASK_LABELS)[0], label="O que variar")
                    lo_itens = gr.Textbox(
                        label="Itens — um por linha", lines=6,
                        placeholder="biquíni de praia vermelho\nterno social preto\n"
                                    "vestido de verão floral\njaqueta de couro")
                    lo_keep = gr.Checkbox(value=True, label="🎯 Preservar a pessoa")

                    with gr.Accordion("Ajustes avançados", open=False):
                        lo_model = gr.Dropdown(choices=list(EDIT_MODEL_LABELS),
                                               value=list(EDIT_MODEL_LABELS)[0], label="Modelo")
                        lo_steps = gr.Slider(10, 60, value=p2p.MODELS[p2p.DEFAULT_MODEL].default_steps,
                                             step=1, label="Passos")
                        lo_guidance = gr.Slider(1.0, 10.0,
                                                value=p2p.MODELS[p2p.DEFAULT_MODEL].default_guidance,
                                                step=0.5, label="Aderência")
                        lo_seed = gr.Number(value=42, label="Seed (igual p/ todos)", precision=0)
                        lo_lowvram = gr.Checkbox(value=False, label="Low-VRAM")
                        lo_melhorar = gr.Checkbox(value=False,
                                                  label="✨ Melhorar qualidade (mais lento por item)")
                        lo_escala = gr.Radio(choices=[1, 2, 4], value=2, label="Upscale (×)")
                        lo_checkid = gr.Checkbox(value=True, label="🔎 Mostrar identidade de cada um")

                    lo_botao = gr.Button("🗂️ Gerar lote", variant="primary")

                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    lo_gallery = gr.Gallery(label="Resultados", columns=3, height=420)
                    lo_status = gr.Textbox(label="Status", interactive=False, lines=4)

            lo_botao.click(lote,
                           inputs=[lo_imagem, lo_task, lo_itens, lo_keep, lo_model,
                                   lo_steps, lo_guidance, lo_seed, lo_lowvram,
                                   lo_melhorar, lo_escala, lo_checkid],
                           outputs=[lo_gallery, lo_status], api_name="lote")

        # ---------------- Aba 5: Histórico ----------------
        with gr.Tab("🕘 Histórico", id="historico"):
            gr.Markdown(
                "> 🕘 Tudo que você gerou (entre sessões) está aqui — salvo em "
                "`outputs/`. Clique numa imagem para selecioná-la e então "
                "**Editar** ou **Animar**. Os vídeos ficam na lista ao lado.",
                elem_classes=["notice"])
            with gr.Row():
                with gr.Column(scale=2, elem_classes=["work-panel"]):
                    hi_gallery = gr.Gallery(value=history.images(), label="Imagens",
                                            columns=5, height=420)
                    hi_selected = gr.Textbox(visible=False)
                    with gr.Row():
                        hi_refresh = gr.Button("🔄 Atualizar")
                        hi_to_edit = gr.Button("🖼️ Editar a selecionada")
                        hi_to_video = gr.Button("🎬 Animar a selecionada")
                    hi_status = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column(scale=1, elem_classes=["work-panel"]):
                    hi_videos = gr.Dropdown(choices=history.videos(), label="Vídeos gerados")
                    hi_player = gr.Video(label="Pré-visualizar vídeo", height=300)

            def _hi_select(evt: gr.SelectData):
                try:
                    v = evt.value
                    path = v.get("image", {}).get("path") if isinstance(v, dict) else v
                    return path or "", f"Selecionada: {Path(path).name}" if path else ""
                except Exception:
                    return "", ""

            def _hi_refresh():
                return (gr.update(value=history.images()),
                        gr.update(choices=history.videos()),
                        "🔄 Atualizado.")

            hi_gallery.select(_hi_select, inputs=None, outputs=[hi_selected, hi_status],
                              api_name=False)
            hi_refresh.click(_hi_refresh, outputs=[hi_gallery, hi_videos, hi_status],
                             api_name=False)
            hi_videos.change(lambda p: p, inputs=hi_videos, outputs=hi_player, api_name=False)

    # --- Fluxo entre abas: editar -> animar / encadear edições ---
    def _enviar_para_video(edited_path):
        if not edited_path:
            return gr.update(), gr.update(), gr.update()
        # joga a imagem editada na entrada do vídeo e pula para a aba de vídeo.
        return (gr.update(value=edited_path), gr.Tabs(selected="video"),
                "✅ Imagem editada carregada na aba de vídeo — escreva o prompt e gere.")

    def _reusar_como_entrada(edited_path):
        if not edited_path:
            return gr.update(), "⚠️ Gere uma edição primeiro."
        return gr.update(value=edited_path), "♻️ Resultado virou a nova entrada — edite de novo (ex.: troque o fundo agora)."

    ed_to_video.click(_enviar_para_video, inputs=ed_saida,
                      outputs=[imagem, tabs, status], api_name=False)
    ed_reuse.click(_reusar_como_entrada, inputs=ed_saida,
                   outputs=[ed_imagem, ed_status], api_name=False)
    st_to_video.click(_enviar_para_video, inputs=st_saida,
                      outputs=[imagem, tabs, status], api_name=False)

    # Histórico -> mandar a imagem selecionada para Editar / Vídeo.
    def _hist_para_editar(sel):
        if not sel:
            return gr.update(), gr.Tabs(), "⚠️ Selecione uma imagem no histórico."
        return (gr.update(value=sel), gr.Tabs(selected="editar"),
                "✅ Carregada na aba Editar foto.")

    def _hist_para_video(sel):
        if not sel:
            return gr.update(), gr.Tabs(), "⚠️ Selecione uma imagem no histórico."
        return (gr.update(value=sel), gr.Tabs(selected="video"),
                "✅ Carregada na aba de vídeo — escreva o prompt e gere.")

    hi_to_edit.click(_hist_para_editar, inputs=hi_selected,
                     outputs=[ed_imagem, tabs, hi_status], api_name=False)
    hi_to_video.click(_hist_para_video, inputs=hi_selected,
                      outputs=[imagem, tabs, hi_status], api_name=False)


if __name__ == "__main__":
    env_port = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT")
    port = int(env_port) if env_port else 7860
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=port,
        show_error=True,
        css=APP_CSS,
        allowed_paths=[str(ASSETS)],
    )
