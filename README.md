# photo2video — foto → vídeo local com Wan2.2

Ferramenta de linha de comando que transforma **uma foto + um prompt** em um
**vídeo curto**, rodando localmente os modelos abertos **Wan2.2** (Wan-AI / Alibaba).
É um wrapper fino sobre o repositório oficial
[`Wan-Video/Wan2.2`](https://github.com/Wan-Video/Wan2.2) que cuida da detecção
de hardware, escolha de modelo, download do checkpoint e montagem do comando.

## ⚠️ Sobre hardware (importante)

Os modelos Wan2.2 são feitos para **GPU NVIDIA com CUDA**. Internamente eles
fixam o device em `cuda:{id}` (ver `wan/textimage2video.py`, `wan/image2video.py`),
então **a geração de verdade exige uma máquina com GPU NVIDIA**.

**Esta máquina é de desenvolvimento e só tem GPU Intel integrada (sem CUDA).**
Aqui o `photo2video.py` roda em **modo validação / dry-run**: ele valida tudo e
imprime o comando exato a executar na máquina com GPU. Para forçar execução em
CPU use `--force` (na prática é inviável — levaria horas e o modelo nem assume CPU).

### Modelos suportados (foto → vídeo)

| Modelo      | Task        | VRAM aprox.                          | Observação                    |
|-------------|-------------|--------------------------------------|-------------------------------|
| `ti2v-5B`   | `ti2v-5B`   | ~8–10GB (FP8), 24GB+ rec. (FP16 ~27GB) | **Default** — mais leve, roda em 1 GPU consumer (ex: RTX 4090) |
| `i2v-A14B`  | `i2v-A14B`  | muito maior (alta VRAM / multi-GPU)  | Mais qualidade, mais pesado   |

## Estrutura do projeto

```
IA_Local/
├── photo2video.py        # o wrapper CLI (ponto de entrada)
├── requirements_cpu.txt  # deps para dev/teste em CPU (sem flash_attn)
├── README.md             # este arquivo
├── venv_wan/             # virtualenv (Python 3.12)
└── Wan2.2/               # repo oficial clonado (+ patch de CPU em wan/modules/t5.py)
```

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
- `wan/configs/__init__.py` — adicionados tamanhos pequenos (`256*256`, `384*384`,
  `512*512`) ao `ti2v-5B` **apenas para smoke test em CPU** (o modelo é treinado em
  720p; baixa resolução serve só para validar o pipeline, a qualidade cai muito).

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
| `-o, --output`       | Arquivo `.mp4` de saída (default: `saida.mp4`)              |
| `--model`            | `ti2v-5B` (default) ou `i2v-A14B`                           |
| `--size`             | Área do vídeo, ex: `1280*704` (default depende do modelo)   |
| `--frames`/`--steps`/`--seed` | Controles de geração                              |
| `--device`           | `auto` (default), `cuda` ou `cpu`                           |
| `--download`         | Baixa o checkpoint se faltar, antes de gerar                |
| `--download-only`    | Só baixa o checkpoint e sai                                 |
| `--dry-run`          | Valida e imprime o comando, sem executar                    |
| `--force`            | Tenta executar mesmo sem CUDA (inviável em CPU)             |

Flags não cobertas podem ser repassadas cruas ao `generate.py` após `--`.

## Deploy em GPU (produção)

1. Copie este projeto (ou só `photo2video.py` + o repo `Wan2.2/`) para a máquina com GPU NVIDIA.
2. Crie o venv e instale PyTorch **CUDA** + `requirements_cpu.txt` + `flash-attn`.
3. Baixe o checkpoint: `python photo2video.py --model ti2v-5B --download-only`.
4. Gere: `python photo2video.py -i foto.jpg -p "..." -o saida.mp4`.

Referência de tempo: no RTX 4090, o `ti2v-5B` gera ~25 frames a 768×512 em ~4–5 min (FP16, 30 passos).
