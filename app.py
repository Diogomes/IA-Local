#!/usr/bin/env python3
"""Interface web (Gradio) do photo2video — foto + prompt -> vídeo com Wan2.2.

Envie uma foto, escreva um prompt e gere um vídeo curto. A fidelidade à pessoa
vem do próprio I2V do Wan (usa a foto como quadro inicial).

QUALIDADE: o ti2v-5B foi treinado em 720p com ~40-50 passos. Resoluções baixas
(256/384) e poucos passos servem só para TESTAR o pipeline — o resultado fica
borrado/"flashes". Para um vídeo de verdade use 512px+ e 30+ passos (lento na
CPU; rápido numa GPU).

Cada geração chama o generate.py oficial como UM subprocesso, então a memória
(T5 ~11GB + DiT ~10GB) é toda liberada ao terminar.

Rodar:  venv_wan/bin/python app.py     (abre em http://127.0.0.1:7860)
"""

from __future__ import annotations

import re
import subprocess
import uuid
import os
from pathlib import Path

import gradio as gr

from photo2video import FPS_DEFAULT, WAN_REPO, generation_command, has_cuda

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "gigaverse3d-logo.png"

CUDA = has_cuda()

# Rótulo amigável -> valor de --size. Áreas pequenas = só teste; 720p = qualidade.
RESOLUCOES = {
    "256px — só teste (qualidade ruim)": "256*256",
    "384px — teste melhor (ainda fraco)": "384*384",
    "512px — qualidade ok (lento em CPU)": "512*512",
    "720p — qualidade boa (ideal em GPU)": "1280*704",
}

# preset -> (resolução, passos, duração s, guidance)
PRESETS = {
    "Teste rápido (CPU)": ("256px — só teste (qualidade ruim)", 8, 2.0, 5.0),
    "Qualidade média": ("512px — qualidade ok (lento em CPU)", 30, 3.0, 5.0),
    "Alta qualidade (GPU)": ("720p — qualidade boa (ideal em GPU)", 40, 5.0, 5.0),
}

PROMPT_EXEMPLO = "a pessoa sorri suavemente e acena para a câmera, luz natural, movimento sutil, alta qualidade"


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


def gerar(imagem_path, prompt, resolucao_label, passos, duracao, guidance,
          progress=gr.Progress()):
    if not imagem_path:
        yield None, "⚠️ Envie uma foto primeiro."
        return
    if not prompt or not prompt.strip():
        yield None, "⚠️ Escreva um prompt descrevendo o movimento/cena."
        return

    size = RESOLUCOES.get(resolucao_label, "256*256")
    frames = frames_para_duracao(float(duracao))
    saida = OUTPUTS / f"video_{uuid.uuid4().hex[:8]}.mp4"
    # Para resoluções baixas o próprio Wan recomenda shift=3.0.
    w, h = (int(x) for x in size.split("*"))
    shift = 3.0 if w * h <= 480 * 480 else 5.0

    cmd, _ = generation_command(
        image=imagem_path, prompt=prompt.strip(), size=size, frame_num=frames,
        steps=int(passos), output=saida, guide_scale=float(guidance),
        shift=shift, cuda=CUDA)

    yield None, (f"⏳ Iniciando… {size}, {frames} frames (~{frames/FPS_DEFAULT:.1f}s), "
                 f"{int(passos)} passos.\n"
                 + ("" if CUDA else "Em CPU é LENTO: carregar o modelo já leva ~1 min "
                    "e cada passo demora; resoluções altas podem levar horas."))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(WAN_REPO))

    fase_difusao = False
    passo_re = re.compile(r"(\d+)/(\d+)\s*\[")
    for linha in proc.stdout:
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
        yield str(saida), f"✅ Pronto! Vídeo de ~{frames/FPS_DEFAULT:.1f}s ({size})."
    else:
        yield None, ("❌ A geração falhou (código %s). Em CPU a causa mais comum é "
                     "falta de memória (OOM). Tente resolução/passos/duração menores, "
                     "ou rode numa máquina com GPU." % proc.returncode)


AVISO_QUALIDADE = (
    "> ⚠️ **Qualidade x velocidade:** o modelo foi treinado em **720p com ~40 passos**. "
    "Resoluções baixas (256/384) e poucos passos servem só para *testar* — saem "
    "borradas/'flashes de luz'. Para um vídeo nítido e fiel ao prompt use **512px+ e "
    "30+ passos**. Na CPU isso é lento (pode levar horas); numa **GPU** sai em minutos.")


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
    gr.Markdown(AVISO_QUALIDADE, elem_classes=["notice"])

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["work-panel"]):
            imagem = gr.Image(type="filepath", label="Foto de entrada", height=300)
            prompt = gr.Textbox(label="Prompt", lines=3,
                                placeholder=PROMPT_EXEMPLO, value=PROMPT_EXEMPLO)
            preset = gr.Radio(choices=list(PRESETS) + ["Personalizado"],
                              value="Teste rápido (CPU)", label="Preset")

            with gr.Accordion("Ajustes avançados", open=False):
                resolucao = gr.Dropdown(choices=list(RESOLUCOES),
                                        value=PRESETS["Teste rápido (CPU)"][0],
                                        label="Resolução")
                passos = gr.Slider(4, 60, value=8, step=1,
                                   label="Passos de difusão (↑ = melhor e mais lento)")
                duracao = gr.Slider(1.0, 10.0, value=2.0, step=0.5,
                                    label="Duração (segundos)")
                guidance = gr.Slider(1.0, 12.0, value=5.0, step=0.5,
                                     label="Guidance (aderência ao prompt)")

            botao = gr.Button("🎬 Gerar vídeo", variant="primary")

        with gr.Column(scale=1, elem_classes=["work-panel"]):
            video = gr.Video(label="Resultado", height=360)
            status = gr.Textbox(label="Status", interactive=False, lines=5)

    preset.change(aplicar_preset, inputs=preset,
                  outputs=[resolucao, passos, duracao, guidance], api_name=False)
    botao.click(gerar,
                inputs=[imagem, prompt, resolucao, passos, duracao, guidance],
                outputs=[video, status], api_name="gerar")


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
