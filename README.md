# DGX Spark LLM + vLLM Installer

Interactive Linux installer for **NVIDIA DGX Spark (GB10)**. It pulls a Blackwell-ready vLLM image from NVIDIA NGC, lets you download a model from Hugging Face *or* point at a local checkpoint, and starts a local **OpenAI-compatible** HTTP endpoint.

```text
./install.sh
    → TUI wizard
        → check GB10 / Docker / CUDA
        → pick Hugging Face model or local folder
        → docker run nvcr.io/nvidia/vllm  (vllm serve)
        → http://HOST:8000/v1/chat/completions
```

## Requirements

- DGX Spark (ARM64 + Blackwell GB10) with DGX OS / Ubuntu
- CUDA 13.x (the NGC container also brings CUDA)
- Docker + NVIDIA Container Toolkit
- Python 3.10+ (3.12 preferred)
- Network access to `nvcr.io` and `huggingface.co`
- Hugging Face token for gated models ([create a token](https://huggingface.co/settings/tokens))

## Install

Copy or clone this repo onto the Spark. Do not copy a `.venv` created on a Mac or PC.

```bash
chmod +x install.sh
./install.sh
```

`install.sh` creates `.venv`, installs Python deps, puts `sparkctl` on `~/.local/bin`, and opens the TUI.

```bash
./install.sh --bootstrap-only   # venv + sparkctl only, no TUI
```

If `docker ps` says permission denied:

```bash
sudo usermod -aG docker "$USER"
newgrp docker    # or log out and back in
./install.sh
```

## Wizard flow

1. **Hardware check** — OS, aarch64, `nvidia-smi`, Docker, NVIDIA runtime
2. **Tokens** — Hugging Face (gated models) and optional NGC API key (`nvcr.io` 401)
3. **Model**
   - Spark-validated catalog (from the [NVIDIA vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md))
   - Search Hugging Face
   - Paste `org/model`
   - Local folder (`config.json` + `.safetensors` / `.bin` already on the Spark)
4. **Serve settings** — port, served name, GPU memory util, start on boot
5. **Download + launch** — pull `nvcr.io/nvidia/vllm:26.05.post1-py3`, start the container, wait on `/health`

Default first-run catalog pick is `Qwen/Qwen2.5-Math-1.5B-Instruct` (small, fast smoke test). NVFP4 NVIDIA checkpoints are preferred for GB10 throughput.

## Endpoint

Once the wizard finishes:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness |
| GET | `/v1/models` | listed model id |
| POST | `/v1/chat/completions` | chat (request in / response out) |

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spark-llm",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Use the **served model name** shown at the end of the wizard (not necessarily `spark-llm`) as `"model"`.

OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="spark-llm",
    messages=[{"role": "user", "content": "Hello"}],
))
```

## `sparkctl` after install

```bash
sparkctl wizard    # run the installer again
sparkctl status    # container + endpoint
sparkctl logs      # vLLM logs (--tail 80)
sparkctl test      # one chat completion
sparkctl stop      # docker rm -f spark-vllm
sparkctl start     # start again from ~/.config/sparkctl/config.json
```

From the repo without PATH: `./sparkctl.sh status`

## How vLLM is launched

The container is started with Spark-oriented flags:

- `--gpus all --ipc=host --network host`
- `--ulimit memlock=-1 --ulimit stack=67108864`
- Hugging Face cache mounted at `/root/.cache/huggingface`
- Local / downloaded weights mounted at `/models/current` (read-only)
- `--gpu-memory-utilization` (default `0.80`)
- `--restart unless-stopped` when “start on boot” is enabled

Override the image with:

```bash
export SPARKCTL_VLLM_IMAGE=nvcr.io/nvidia/vllm
export SPARKCTL_VLLM_TAG=26.05.post1-py3
```

(Tag is stored in `~/.config/sparkctl/config.json` after the first run.)

## Local models

“Upload from this computer” means the files are **already on the Spark**. Copy them first:

```bash
scp -r ./my-instruct-model spark:/home/<user>/models/my-instruct-model
```

Then choose **Local folder** in the wizard and pick that directory. It must look like a Hugging Face Transformers checkpoint (`config.json` + shards).

## Troubleshooting

**`docker pull nvcr.io/nvidia/vllm` → 401**  
Paste an NGC API key in the wizard, or:

```bash
docker login nvcr.io
# username: $oauthtoken
# password: NGC API key from https://ngc.nvidia.com
```

**Gated Hugging Face model (401 / 403)**  
Accept the model license on the Hub, then paste a token with read access.

**Container exits while loading**  
`sparkctl logs` — often OOM. Lower GPU memory utilization or pick a smaller / NVFP4 checkpoint.

**Not detected as GB10**  
The wizard still continues if Docker + GPU checks pass, with a warning.

## Layout

```text
install.sh                 bootstrap + launch TUI
sparkctl.sh                repo-local CLI wrapper
sparkctl/                  Python package (TUI, hardware, HF, vLLM)
packaging/spark-vllm.service
requirements.txt
```
