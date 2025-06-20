import json
import os
import torch
import wandb
import random
import argparse

from safetensors import safe_open

from transformers.models.llama.configuration_llama import LlamaConfig

from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments

from modules.model.llama_eagle import LlamaForCausalLMEagle
from modules.data.data import (
    EagleLocalDataset,
    DataCollatorWithPadding,
    AddUniformNoise,
    list_local_files,
)
from modules.trainer.trainer import EagleTrainer

run_name = "06-20-2025-Qwen2.5-7B-Instruct-EAGLE"
wandb.init(project="BaldEagle", mode="offline", name=run_name)
wandb_run_name = wandb.run.name

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=os.environ["MODEL_PATH"])
parser.add_argument("--sharegpt-datapaths", type=str, default=os.environ["SHAREGPT_DATAPATHS"])
parser.add_argument("--ultra-chat-datapaths", type=str, default=os.environ["ULTRACHAT_DATAPATHS"])
parser.add_argument("--output-dir", type=str, default=f"./hf_trainer_output_dir/{wandb_run_name}")
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

model_path = args.model_path
sharegpt_datapaths = args.sharegpt_datapaths
ultra_chat_datapaths = args.ultra_chat_datapaths
hf_repo = "baseten-admin/qwen2-5-eagle-test"

# -------------------------------- Load original Llama weights --------------------------------

with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
    lm_head_path = index_json["weight_map"]["lm_head.weight"]

with safe_open(os.path.join(model_path, emb_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("model.embed_tokens.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]

with safe_open(os.path.join(model_path, lm_head_path), framework="pt", device="cpu") as f:
    lm_head_weights = f.get_slice("lm_head.weight")[:, :]


# -------------------------------- Create draft model + tokenizer + head --------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model_args = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_dim,
    intermediate_size=12288,
    num_hidden_layers=1,
    bos_token_id=128000,
    eos_token_id=[128001, 128008, 128009],
    num_key_value_heads=28,
    num_attention_heads=28,
    tie_word_embeddings=False,
)

draft_model = LlamaForCausalLMEagle(model_args)
draft_model.load_embedding_weights(tensor)
draft_model.to("cuda:0")
draft_model.embed_tokens.weight.requires_grad = False

# Load head
head = torch.nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False)
with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
with safe_open(os.path.join(model_path, head_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("lm_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].float()

head.weight.data = tensor
head.to("cuda:0")
head.eval()

# -------------------------------- Load data --------------------------------

sharegpt_datapaths = list_local_files(sharegpt_datapaths)
ultra_chat_datapaths = list_local_files(ultra_chat_datapaths)

combined_data_paths = (
    sharegpt_datapaths[: int(len(sharegpt_datapaths) * 0.95)] + ultra_chat_datapaths
)
random.Random(42).shuffle(combined_data_paths)
eval_data_paths = sharegpt_datapaths[int(len(sharegpt_datapaths) * 0.95) :][:100]

eagle_train_dataset = EagleLocalDataset(
    combined_data_paths, transform=AddUniformNoise(std=0.5)
)
eagle_test_dataset = EagleLocalDataset(eval_data_paths)

eagle_collator = DataCollatorWithPadding()

# -------------------------------- Train --------------------------------

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    gradient_accumulation_steps=16,
    fsdp=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    bf16=True,
    fp16=False,
    dataloader_num_workers=4,
    warmup_ratio=0.01,
    learning_rate=1e-4,  # 1e-3
    lr_scheduler_type="constant",  # Placeholder, we override it in the trainer
    max_grad_norm=0.5,  # 1
    adam_beta1=0.9,  # 0.9
    adam_beta2=0.95,  # 0.999
    weight_decay=1e-2,
    eval_strategy="steps",
    logging_steps=32,
    eval_steps=64,
    save_strategy="steps",
    save_steps=0.1,  # saves every 10% of training
    save_total_limit=3,
)

trainer = EagleTrainer(
    model=draft_model,
    head=head,
    args=training_args,
    train_dataset=eagle_train_dataset,
    eval_dataset=eagle_test_dataset,
    data_collator=eagle_collator,
    min_lr_ratio=0.5,  # Custmer lr scheduler param
)

trainer.train()
trainer.push_to_hub(hf_repo)
