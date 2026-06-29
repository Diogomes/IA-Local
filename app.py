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
from pathlib import Path

import gradio as gr

from photo2video import FPS_DEFAULT, WAN_REPO, generation_command, has_cuda

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

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


def gerar(imagem_path, prompt, resolucao_label, passos, duracao, guidance,
          manter_pessoa, negativo_extra, progress=gr.Progress()):
    if not imagem_path:
        yield None, "⚠️ Envie uma foto primeiro."
        return
    if not prompt or not prompt.strip():
        yield None, "⚠️ Escreva um prompt descrevendo o movimento/cena."
        return

    size = RESOLUCOES.get(resolucao_label, "256*256")
    frames = frames_para_duracao(float(duracao))
    saida = OUTPUTS / f"video_{uuid.uuid4().hex[:8]}.mp4"
    # Em resoluções pequenas o Wan recomenda shift=3.0; em 480p+ usa 5.0.
    w, h = (int(x) for x in size.split("*"))
    shift = 3.0 if w * h <= 512 * 512 else 5.0

    cmd, _ = generation_command(
        image=imagem_path, prompt=prompt.strip(), size=size, frame_num=frames,
        steps=int(passos), output=saida, guide_scale=float(guidance),
        shift=shift, cuda=CUDA, keep_identity=bool(manter_pessoa),
        negative_prompt=(negativo_extra or "").strip() or None)

    yield None, (f"⏳ Iniciando… {size}, {frames} frames (~{frames/FPS_DEFAULT:.1f}s), "
                 f"{int(passos)} passos, guidance {guidance}.\n"
                 + ("🎯 Preservação da pessoa: LIGADA.\n" if manter_pessoa else "")
                 + ("" if CUDA else "Em CPU é LENTO: carregar o modelo já leva ~1 min "
                    "e cada passo demora; resoluções altas podem levar horas."))

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


with gr.Blocks(title="photo2video — Wan2.2") as demo:
    gr.Markdown(
        "# 🎞️ photo2video — foto → vídeo realista (Wan2.2)\n"
        "Envie uma **foto**, escreva um **prompt** com o movimento/cena desejado e "
        "gere um vídeo curto. A pessoa da foto é **preservada**.\n\n" + _GPU_TXT)
    gr.Markdown(AVISO_QUALIDADE)

    with gr.Row():
        with gr.Column(scale=1):
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

            botao = gr.Button("🎬 Gerar vídeo", variant="primary")

        with gr.Column(scale=1):
            video = gr.Video(label="Resultado", height=360)
            status = gr.Textbox(label="Status", interactive=False, lines=6)

    preset.change(aplicar_preset, inputs=preset,
                  outputs=[resolucao, passos, duracao, guidance], api_name=False)
    botao.click(gerar,
                inputs=[imagem, prompt, resolucao, passos, duracao, guidance,
                        manter_pessoa, negativo_extra],
                outputs=[video, status], api_name="gerar")


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
