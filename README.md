# photo2video — foto → vídeo local com Wan2.2

Ferramenta de linha de comando que transforma **uma foto + um prompt** em um
**vídeo curto**, rodando localmente os modelos abertos **Wan2.2** (Wan-AI / Alibaba).
É um wrapper fino sobre o repositório oficial
[`Wan-Video/Wan2.2`](https://github.com/Wan-Video/Wan2.2) que cuida da detecção
de hardware, escolha de modelo, download do checkpoint e montagem do comando.

## ⚠️ Sobre hardware (importante)

A **geração em alta qualidade roda numa GPU NVIDIA**. A máquina de desenvolvimento
aqui só tem GPU Intel integrada (sem CUDA) — nela o `photo2video.py` roda em CPU
(lento, só para *testar* o pipeline) ou em dry-run. A geração de verdade (540p/720p)
é feita no **PC com a RTX 5070 Ti 16GB** — veja
[Deploy no PC com RTX 5070 Ti (Windows)](#deploy-no-pc-com-rtx-5070-ti-windows).

> 🟢 **RTX 50 (Blackwell) — leia isto:** a 5070 Ti usa a arquitetura **sm_120** e só
> funciona com **PyTorch CUDA 12.8 (cu128)**. Os builds cu121/cu124 falham com
> *"no kernel image available for execution on the device"*. O `setup_gpu_windows.ps1`
> já instala o build certo.

### Modelos suportados (foto → vídeo)

| Modelo      | Task        | VRAM aprox.                          | Observação                    |
|-------------|-------------|--------------------------------------|-------------------------------|
| `ti2v-5B`   | `ti2v-5B`   | ~12–16GB com `--offload_model` (bf16) | **Default** — cabe na RTX 5070 Ti 16GB; 720p OK |
| `i2v-A14B`  | `i2v-A14B`  | muito maior (alta VRAM / multi-GPU)  | Mais qualidade, **não recomendado em 16GB** (lento/OOM) |

## Estrutura do projeto

```
IA_Local/
├── app.py                # Interface web (Gradio): abas "Foto → Vídeo" e "Editar foto"
├── photo2video.py        # wrapper CLI foto -> vídeo (Wan2.2)
├── photo2photo.py        # edição de foto (roupa/fundo/corpo) mantendo a pessoa
├── requirements_cpu.txt  # deps para dev/teste em CPU (sem flash_attn)
├── README.md             # este arquivo
├── outputs/              # vídeos gerados pela UI
├── venv_wan/             # virtualenv (Python 3.12)
└── Wan2.2/               # repo oficial clonado (+ patches de CPU)
```

## Interface web (recomendada)

```bash
venv_wan/bin/python app.py     # abre em http://127.0.0.1:7860
```

Na página: arraste uma **foto**, escreva um **prompt**, escolha um preset
(Rápido / Equilibrado / Alto-GPU) ou ajuste resolução/passos/duração à mão, e
clique em **Gerar vídeo**. O rosto/aparência da(s) pessoa(s) é preservado porque
a foto vira o quadro inicial do vídeo. Cada geração roda como um processo
separado (libera toda a RAM ao terminar).

## Setup (já feito nesta máquina de dev)

```bash
# 1) venv com Python 3.12 (PyTorch ainda não suporta 3.14)
python3.12 -m venv venv_wan
source venv_wan/bin/activate

# 2) PyTorch — CPU para dev:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#    ...ou CUDA na máquina de produção (escolha a versão CUDA correta):
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3) demais dependências (omite flash_attn de propósito; veja nota abaixo)
pip install -r requirements_cpu.txt
```

### Nota sobre `flash_attn`

O `requirements.txt` oficial pede `flash_attn`, que **só compila com CUDA**.
Em CPU ele é omitido — o Wan cai automaticamente no fallback
`torch.scaled_dot_product_attention`. **Na máquina com GPU**, instale-o para
desempenho ótimo:

```bash
pip install flash-attn --no-build-isolation
```

### Patches aplicados ao repo oficial (para suporte a CPU)

Todos marcados com `PATCH (photo2video)` e mantêm o comportamento original em GPU
(quando `torch.cuda.is_available()`):

- `wan/modules/t5.py` — default `device=torch.cuda.current_device()` (avaliado no
  import) → `None`, resolvido em runtime (GPU se houver, senão CPU).
- `wan/textimage2video.py` — `self.device` agora cai em `cpu` sem CUDA; os
  `autocast('cuda', …)` usam device dinâmico (`cpu` casta as entradas para bf16,
  batendo com os pesos convertidos).
- `wan/modules/model.py` — todos os `autocast('cuda', …)` usam device dinâmico
  (preserva os blocos forçados a fp32 — rope/norm — também em CPU).
- `generate.py` — `torch.cuda.synchronize()` final só roda se houver CUDA.
- `wan/configs/__init__.py` — adicionados:
  - tamanhos intermediários `960*544` / `544*960` (≈540p) e habilitado
    `832*480` / `480*832` (≈480p) para o `ti2v-5B` — **qualidade boa que cabe em
    GPUs de 12–16GB** (entre o teste de CPU e o 720p cheio).
  - tamanhos pequenos (`256*256`, `384*384`, `512*512`) **apenas para smoke test
    em CPU** (o modelo é treinado em 720p; baixa resolução só valida o pipeline).
- `generate.py` — adicionada a flag `--negative_prompt` (passada como `n_prompt`
  ao pipeline ti2v/i2v). Vazia = usa o negativo padrão do modelo. Usada pelo
  wrapper para reforçar a preservação da identidade.

### Rodando em CPU (lento — só para teste)

Sem GPU, o `photo2video.py` executa de verdade, mas é **lento e pesado em RAM**.
Use resolução pequena, poucos passos e poucos frames:

```bash
python photo2video.py \
  -i Wan2.2/examples/i2v_input.JPG \
  -p "a pessoa sorri e acena" \
  --size 256*256 --frames 13 --steps 6 \
  -o teste_cpu.mp4 --device cpu
```

Para gerar em qualidade real (720p, ~5s, 40-50 passos), use uma máquina com GPU.

## Uso

```bash
source venv_wan/bin/activate

# Validar / dry-run (qualquer máquina, sem precisar do checkpoint):
python photo2video.py -i foto.jpg -p "a pessoa sorri e acena" --dry-run

# Baixar o checkpoint do modelo leve (vários GB):
python photo2video.py --model ti2v-5B --download-only

# Gerar de verdade (máquina com GPU NVIDIA):
python photo2video.py -i foto.jpg -p "a pessoa sorri e acena" -o saida.mp4
```

### Opções principais

| Flag                 | Descrição                                                   |
|----------------------|-------------------------------------------------------------|
| `-i, --image`        | Foto de entrada (obrigatório p/ gerar)                      |
| `-p, --prompt`       | Descrição do movimento/cena (obrigatório p/ gerar)          |
| `-n, --negative-prompt` | Termos extras a EVITAR (somados ao negativo padrão)      |
| `--no-keep-identity` | Desliga o reforço de preservação da pessoa (rosto)          |
| `-o, --output`       | Arquivo `.mp4` de saída (default: `saida.mp4`)              |
| `--model`            | `ti2v-5B` (default) ou `i2v-A14B`                           |
| `--size`             | Área do vídeo, ex: `1280*704` (default depende do modelo)   |
| `--frames`/`--steps`/`--seed` | Controles de geração                              |
| `--device`           | `auto` (default), `cuda` ou `cpu`                           |
| `--download`         | Baixa o checkpoint se faltar, antes de gerar                |
| `--download-only`    | Só baixa o checkpoint e sai                                 |
| `--dry-run`          | Valida e imprime o comando, sem executar                    |

Flags não cobertas podem ser repassadas cruas ao `generate.py` após `--`.

## Deploy no PC com RTX 5070 Ti (Windows)

Esse é o caminho para gerar vídeos **de verdade**, em alta qualidade.

### Passo a passo

1. **Pré-requisitos no PC da GPU:**
   - Driver NVIDIA recente (Game Ready ou Studio) com suporte a CUDA 12.8.
   - **Python 3.12 (64-bit)** — baixe em
     <https://www.python.org/downloads/release/python-3120/> e marque
     *"Add python.exe to PATH"*. (PyTorch ainda não tem wheels para 3.13/3.14.)

2. **Copie o projeto** para o PC da GPU (a pasta `IA_Local/` inteira, incluindo
   `Wan2.2/` com os patches; **não** precisa copiar o `venv_wan/`).

3. **Rode o setup** (PowerShell, dentro da pasta do projeto):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_gpu_windows.ps1
   ```
   Ele cria o `venv_wan`, instala o **PyTorch cu128** (obrigatório p/ Blackwell),
   as dependências (`requirements_cuda.txt`), confere se a GPU foi detectada e
   oferece baixar o checkpoint `ti2v-5B` (~16GB).

4. **Abra a interface web:**
   ```
   run_ui_windows.bat
   ```
   (ou `venv_wan\Scripts\python app.py`) — abre em <http://127.0.0.1:7860>.

5. Envie a foto, escreva o prompt, deixe **"Preservar a pessoa"** ligado, escolha
   o preset **"Máxima qualidade (GPU)"** e gere.

### Pela linha de comando (Windows)

```powershell
venv_wan\Scripts\python photo2video.py -i foto.jpg -p "a pessoa vira a cabeça e sorri" -o saida.mp4
```

### Dicas de VRAM (16GB)

- **540p (`960*544`)** com 40–50 passos cabe folgado e já fica nítido. **Comece por aqui.**
- **720p (`1280*704`)** é o topo de qualidade; em 16GB pode ficar no limite com 5s.
  Se der **OOM**, reduza a duração (ex.: 3–4s) ou caia para 540p.
- O wrapper sempre usa `--offload_model True` + `--convert_model_dtype` + `--t5_cpu`
  na GPU, que é o que faz o ti2v-5B caber em 16GB.

Referência de tempo: numa GPU desta classe, ~5s a 720p (50 passos) leva
poucos minutos (a primeira geração é mais lenta porque carrega os modelos).

## Editar foto: trocar roupa / fundo / corpo (`photo2photo`)

Além de gerar vídeo, a ferramenta edita uma foto **mantendo a mesma pessoa**,
usando modelos de edição por instrução abertos e gratuitos (via 🤗 diffusers):

| Tarefa | O que faz |
|--------|-----------|
| 👗 **Trocar roupa** | Troca o look (inclui **roupa de praia/biquíni**, social, casual) sem mexer no rosto/fundo |
| 🏞️ **Trocar fundo/cenário** | Coloca a pessoa em outro local (praia, escritório, rua…) preservando-a |
| 🧍 **Recriar corpo inteiro** | A partir de um retrato **só do rosto**, gera a pessoa de corpo inteiro |
| 🧥 **Try-on** | Veste a pessoa com uma peça de uma **2ª foto** de referência |

### Modelos (escolha na aba ou via `--model`)

| Modelo | Repo HF | Licença | Nota |
|--------|---------|---------|------|
| `qwen-edit` *(default)* | `Qwen/Qwen-Image-Edit-2509` | Apache-2.0, **sem gate** | ~20B → em 16GB use `--quantize 4bit` (default) |
| `flux-kontext` | `black-forest-labs/FLUX.1-Kontext-dev` | Não-comercial, **gated** | ~12B; aceite a licença no HF e rode `hf auth login` |

### Uso

Na UI: aba **"🖼️ Editar foto"** → envie a foto, escolha a tarefa, descreva a
mudança e clique em **Editar foto** (a 1ª vez baixa o modelo, vários GB).

Pela CLI (no PC com GPU):
```powershell
venv_wan\Scripts\python photo2photo.py -i foto.jpg --task roupa -d "biquíni de praia azul" -o saida.png
venv_wan\Scripts\python photo2photo.py -i rosto.jpg --task corpo -d "em pé, jeans e camiseta branca"
venv_wan\Scripts\python photo2photo.py -i pessoa.jpg --ref camisa.jpg --task tryon
```

O `--quantize 4bit` (padrão) faz o editor caber em 16GB; se faltar VRAM, some
`--lowvram` (offload sequencial, mais lento). Sem GPU, roda só em `--dry-run`.

> ⚖️ **Uso responsável:** edite fotos suas ou de pessoas que consentiram (moda,
> try-on, restauração). A ferramenta **não** se destina a criar imagens
> sexuais/íntimas de pessoas reais sem consentimento.

## Fidelidade à pessoa (não mudar quem está na foto)

A semelhança vem de três coisas combinadas:

1. **A foto é o quadro inicial** do vídeo (próprio I2V do Wan).
2. **Reforço de identidade nos prompts** — esta ferramenta adiciona, por padrão,
   ao prompt positivo *"mesma pessoa, mesmo rosto, identidade preservada,
   fotorrealista"* e ao negativo termos como *"rosto diferente, troca de
   identidade, deformação facial, morphing"*. Controlado por
   `--no-keep-identity` (CLI) ou pela caixa **"Preservar a pessoa"** (UI).
3. **Boas escolhas de geração:**
   - **Guidance ~5** (o ideal do modelo). Muito alto distorce o rosto; muito baixo ignora o prompt.
   - No prompt, descreva **só o movimento/cena** ("vira a cabeça", "sorri", "anda
     em direção à câmera") — **não** descreva as feições da pessoa, senão o modelo
     tende a "recriar" um rosto.
   - Movimentos **sutis e curtos (~5s)** preservam melhor; clipes longos derivam.
   - Use uma foto **nítida, de frente e bem iluminada** do rosto.
