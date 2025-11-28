"""Modal workflow to fine-tune GPT-OSS-20B (4-bit) on a price prediction dataset.

This script sets up a Modal stub that installs the required dependencies, loads the
quantized GPT-OSS-20B model, and fine-tunes it with QLoRA on the
``costadev00/pricer-data`` dataset. It follows a prompt/target split around the
"Price is $" marker so that the model learns to generate only the price.

Usage (from a machine with Modal credentials configured):
    modal run fine_tune_gpt_oss_modal.py --push-to-hub --repo-id your-username/gpt-oss-pricer
"""

from __future__ import annotations

import os

import modal


stub = modal.Stub("gpt_oss_20b_fine_tune")


image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "torch==2.0.1+cu118",
            "transformers==4.56.0",
            "accelerate==0.21.0",
            "datasets==2.14.4",
            "bitsandbytes==0.41.1",
            "peft==0.5.0",
            "huggingface_hub==0.21.3",
        ],
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
)

volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@stub.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 4,
    secret=modal.Secret.from_name("huggingface-token"),
    volumes={"/root/.cache/huggingface": volume},
)
def train(push_to_hub: bool = False, repo_id: str | None = None):
    """Fine-tune GPT-OSS-20B 4-bit with QLoRA on the price dataset."""

    import torch
    from datasets import load_dataset
    from huggingface_hub import login
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    model_name = "unsloth/gpt-oss-20b-bnb-4bit"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("costadev00/pricer-data")
    train_dataset = dataset["train"]
    test_dataset = dataset.get("test") or dataset["validation"]

    def preprocess_example(example):
        text = example["text"]
        marker = "Price is $"
        if marker not in text:
            return {"input_ids": [], "labels": []}
        split_index = text.index(marker)
        prompt = text[: split_index + len(marker)]
        price_str = text[split_index + len(marker) :]

        prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
        price_tokens = tokenizer(price_str, add_special_tokens=False).input_ids

        input_ids = prompt_tokens + price_tokens
        labels = [-100] * len(prompt_tokens) + price_tokens
        return {"input_ids": input_ids, "labels": labels}

    train_dataset = train_dataset.map(
        preprocess_example, remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        preprocess_example, remove_columns=test_dataset.column_names
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = "/root/model-output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and repo_id:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    sample_prompt = "How much does this cost? Example product... Price is $"
    model.eval()
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=6)
    generated = tokenizer.decode(outputs[0])
    print("Model output:", generated)


@stub.local_entrypoint()
def main(push_to_hub: bool = False, repo_id: str | None = None):
    train.call(push_to_hub=push_to_hub, repo_id=repo_id)
