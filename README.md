# GPT-OSS-20B QLoRA fine-tuning on Modal

This repository contains a Modal workflow for fine-tuning the quantized GPT-OSS-20B model on the [`costadev00/pricer-data`](https://huggingface.co/datasets/costadev00/pricer-data) dataset using the QLoRA recipe. The training job runs on a GPU provided by Modal and saves LoRA adapter weights plus tokenizer artifacts for reuse.

## Prerequisites
- Python 3.10+
- Modal CLI configured and authenticated (`modal token new`)
- A Hugging Face access token stored as a Modal secret named `huggingface-token`
- Optional: a Modal volume named `huggingface-cache` to reuse model downloads (the script will create it if missing)

## Installation
Install the required Python packages (includes CUDA-enabled PyTorch 2.0.1 from the official index):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running training on Modal
The entrypoint `fine_tune_gpt_oss_modal.py` defines a `train` function executed on Modal. Use the CLI to launch fine-tuning:

```bash
modal run fine_tune_gpt_oss_modal.py --push-to-hub --repo-id <your-username/gpt-oss-pricer>
```

Key behaviors:
- Requests any available GPU and a 4-hour timeout.
- Loads `unsloth/gpt-oss-20b-bnb-4bit` with 4-bit quantization.
- Splits each dataset sample at the `"Price is $"` marker so the loss is applied only to the price tokens.
- Attaches LoRA adapters (`r=8`, `alpha=32`, `dropout=0.05`) to all linear layers and trains for 2 epochs with gradient accumulation.
- Saves outputs to `/root/model-output` inside the Modal job. If `--push-to-hub` and `--repo-id` are provided, the model and tokenizer are uploaded to the specified Hugging Face repo.

## Verifying locally (optional)
If you want to dry-run the script without launching a remote job, you can inspect the training flow locally:

```bash
python fine_tune_gpt_oss_modal.py
```

The local entrypoint simply triggers the Modal function; it still requires Modal credentials because the work executes remotely. Use `modal deploy` if you prefer turning this into a deployed function.

## Inference after training
The script prints a short sample generation after training. To use the saved adapters later, load the base quantized model and apply the adapters from the output directory or your Hugging Face repo, then generate text that ends with `"Price is $"` to receive a predicted price.
