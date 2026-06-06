"""
train_lora.py
─────────────
QLoRA fine-tune for LFM2 brain model using HuggingFace Trainer.
Run after generate_finetune_dataset.py.

Requirements:
    pip install transformers datasets peft trl bitsandbytes accelerate

Usage:
    python train_lora.py \
        --base_model ./models/lfm2_brain \
        --train_data train.jsonl \
        --val_data   val.jsonl \
        --output_dir ./models/lfm2_brain_sudo

The output directory contains a LoRA adapter. To merge into a full model:
    python train_lora.py --merge_only \
        --base_model ./models/lfm2_brain \
        --adapter_dir ./models/lfm2_brain_sudo \
        --output_dir ./models/lfm2_brain_merged
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# LFM2 uses a custom model class — keep trust_remote_code=True
from transformers import Lfm2VlForConditionalGeneration


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",  default="./models/lfm2_brain")
    p.add_argument("--train_data",  default="train.jsonl")
    p.add_argument("--val_data",    default="val.jsonl")
    p.add_argument("--output_dir",  default="./models/lfm2_brain_sudo")
    p.add_argument("--adapter_dir", default=None,
                   help="For --merge_only: path to LoRA adapter")
    p.add_argument("--merge_only",  action="store_true")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lora_rank",   type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--max_length",  type=int,   default=512)
    p.add_argument("--use_4bit",    action="store_true", default=False,
                   help="QLoRA: quantise base model to 4-bit (requires GPU with bitsandbytes)")
    return p.parse_args()


# ── Merge adapter into base model ─────────────────────────────────────────────
def merge_and_save(args):
    from peft import PeftModel
    print("Loading base model for merge...")
    base = Lfm2VlForConditionalGeneration.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16,
    )
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    print("Merging weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    tok.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")


# ── Load JSONL ────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Tokenise a single conversation ───────────────────────────────────────────
def tokenise_example(example, tokenizer, max_length):
    """
    Formats messages with the model's chat template, then masks out
    the prompt tokens from the loss (we only train on assistant tokens).
    """
    messages = example["messages"]

    # Full sequence: system + user + assistant
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Prompt only (system + user), so we know where assistant starts
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_ids   = tokenizer(full_text,   max_length=max_length, truncation=True)["input_ids"]
    prompt_ids = tokenizer(prompt_text, max_length=max_length, truncation=True)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Truncate labels to same length as input_ids
    input_ids = full_ids[:max_length]
    labels    = labels[:max_length]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


# ── Main training ─────────────────────────────────────────────────────────────
def train(args):
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = Lfm2VlForConditionalGeneration.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if not args.use_4bit else None,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        # Target the attention projection matrices — standard for decoder LMs
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"Loading train data from {args.train_data}...")
    train_raw = load_jsonl(args.train_data)
    val_raw   = load_jsonl(args.val_data)

    def process(batch):
        results = {"input_ids": [], "labels": [], "attention_mask": []}
        for ex in batch["messages"]:
            # batch["messages"] is a list of lists-of-dicts when using Dataset.map
            tok = tokenise_example({"messages": ex}, tokenizer, args.max_length)
            for k in results:
                results[k].append(tok[k])
        return results

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_raw])
    val_ds   = Dataset.from_list([{"messages": r["messages"]} for r in val_raw])

    train_ds = train_ds.map(
        lambda ex: tokenise_example(ex, tokenizer, args.max_length),
        remove_columns=["messages"]
    )
    val_ds = val_ds.map(
        lambda ex: tokenise_example(ex, tokenizer, args.max_length),
        remove_columns=["messages"]
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,   # keep single-process for MPS/CPU safety
    )

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")
    print("\nTo merge into a full model for inference:")
    print(f"  python train_lora.py --merge_only "
          f"--base_model {args.base_model} "
          f"--adapter_dir {args.output_dir} "
          f"--output_dir {args.output_dir}_merged")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    if args.merge_only:
        merge_and_save(args)
    else:
        train(args)
