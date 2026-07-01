# Runbook — rodar na RTX 5070 Ti (Windows / PowerShell)

Guia para colocar o app para gerar **de verdade na GPU** (Blackwell / sm_120) no
PC com a RTX 5070 Ti. A máquina de desenvolvimento (Fedora, sem GPU) só edita o
código; a validação em CUDA acontece aqui.

> **Causa raiz do "Dispositivo detectado: CPU (lento)"**: o PyTorch instalado não
> tem kernels para a arquitetura Blackwell (`sm_120`). Builds `cu121`/`cu124` (e o
> `+cpu`) falham com *"no kernel image is available for execution on the device"*.
> A correção é o build **CUDA 12.8 (`cu128`)**, que o `setup_gpu_windows.ps1` já
> instala.

Abra o **PowerShell na pasta do projeto**. Faça fase por fase.

## Fase 0 — Diagnóstico (não muda nada)

```powershell
# 0.1 Driver + GPU (tem que listar a 5070 Ti e uma versão de driver)
nvidia-smi

# 0.2 Código atualizado (os scripts cu128/doctor estão na branch feature-doctor)
git fetch origin
git checkout feature-doctor
git pull

# 0.3 Diagnóstico completo (PyTorch/CUDA/Blackwell/modelos/libs)
.\run_doctor_windows.bat
```

Se o `doctor` marcar **FAIL em "Suporte Blackwell"** ou **"CUDA indisponível"** →
siga para a Fase 1. Se estiver tudo OK, pule para a Fase 5.

## Fase 1 — (Re)instalar com suporte Blackwell (cu128)

O script oficial cria o `venv_wan`, instala **torch cu128**, as deps, verifica
`sm_120` e (opcional) baixa os modelos:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_gpu_windows.ps1
```

Se o `venv_wan` já existir com um torch **antigo** (foi ele que instalou sem
`sm_120`), force a reinstalação dentro dele:

```powershell
venv_wan\Scripts\python -m pip uninstall -y torch torchvision torchaudio
venv_wan\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
venv_wan\Scripts\python -m pip install --upgrade "bitsandbytes>=0.46" accelerate transformers diffusers safetensors
```

**Validação estrita** (tem que imprimir `sm_120` e `KERNEL OK`):

```powershell
venv_wan\Scripts\python -c "import torch; al=torch.cuda.get_arch_list(); print('torch',torch.__version__,'cuda',torch.version.cuda); print('arch',al); assert 'sm_120' in al, 'AINDA sem sm_120 - trocar p/ cu129 ou atualizar driver'; x=torch.randn(4096,4096,device='cuda'); y=x@x; torch.cuda.synchronize(); print('KERNEL OK na', torch.cuda.get_device_name(0))"
```

Se der `no kernel image` ou faltar `sm_120` → o driver NVIDIA está velho demais
para CUDA 12.8; atualize o driver (Game Ready/Studio recente) ou troque `cu128`
por `cu129` nos comandos acima.

## Fases 2 e 3 — já estão no código ✅

Nada a instalar. Para referência do que já existe:

- **Device único**: `app.py` (`CUDA = has_cuda()`) e `photo2photo.py`
  (`"cuda" if torch.cuda.is_available() else "cpu"`). Os guards "só roda na GPU"
  só disparam sem CUDA — com a GPU OK, seguem para a geração.
- **VRAM 16 GB**: `--quantize 4bit` (bitsandbytes), offload
  `sequential/model/cuda` e a flag `lowvram` (`photo2photo._place_on_cuda`).
- **Alocação**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` é setado pelo
  próprio `app.py` no boot; `free_vram()` (empty_cache) roda entre gerações.

## Fase 5 — Validação ponta a ponta

**Terminal A** (monitor da GPU, deixe aberto):

```powershell
while ($true) { cls; nvidia-smi; Start-Sleep 1 }
```

**Terminal B**:

```powershell
# smoke test: gera um vídeo mínimo de verdade (confirma kernel + pipeline)
venv_wan\Scripts\python doctor.py --smoke

# abre a interface (cabeçalho deve mostrar "GPU: NVIDIA GeForce RTX 5070 Ti")
venv_wan\Scripts\python app.py
```

No navegador (`http://127.0.0.1:7860`), rode os testes obrigatórios:

1. **Editar foto** com uma foto real → gera (sem a mensagem "só roda na GPU"); o
   `nvidia-smi` mostra VRAM/uso subindo.
2. **Foto → Vídeo** com a mesma foto → produz vídeo, com uso de GPU e tempo em
   minutos (não horas).
3. As duas **em sequência** → a VRAM é liberada entre elas (sem OOM).

Reporte: tempos de geração, pico de VRAM por aba e se houve OOM.

## Decisões (padrões atuais)

- **cu128 vs cu129**: começa em **cu128** (o que o repo usa). Só troca se o driver
  reclamar na Fase 1.
- **Attention**: **SDPA nativo** do PyTorch; `flash-attn` fica de fora de propósito
  (o Wan cai no fallback SDPA, ótimo em CUDA).
- **Fallback CPU**: mantido como **aviso não-bloqueante**.
- **Offload**: para o editor quantizado, tenta `model → cuda → sequential`
  (equilíbrio); use o checkbox **Low-VRAM** se estourar em 720p.
