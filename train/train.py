import json
import os
import torch
import torch.distributed as dist
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

# Initialize wandb only on rank 0 to avoid conflicts
run_name = "06-20-2025-Qwen2.5-7B-Instruct-EAGLE"
if not dist.is_initialized() or dist.get_rank() == 0:
    wandb.init(project="BaldEagle", mode="offline", name=run_name)
    wandb_run_name = wandb.run.name
else:
    wandb_run_name = run_name  # Use static name for non-rank-0 processes

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
    tensor = tensor_slice[:, :hidden_dim].to(torch.bfloat16)

with safe_open(os.path.join(model_path, lm_head_path), framework="pt", device="cpu") as f:
    lm_head_weights = f.get_slice("lm_head.weight")[:, :].to(torch.bfloat16)


# -------------------------------- Create draft model + tokenizer + head --------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model_args = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_dim,
    intermediate_size=12288,
    num_hidden_layers=1,
    bos_token_id=128000,
    eos_token_id=128001,  # Use the first EOS token ID as int
    num_key_value_heads=28,
    num_attention_heads=28,
    tie_word_embeddings=False,
    torch_dtype=torch.bfloat16,
)

draft_model = LlamaForCausalLMEagle(model_args)
draft_model.load_embedding_weights(tensor)
# Ensure all parameters are bfloat16 for FSDP
for param in draft_model.parameters():
    param.data = param.data.to(torch.bfloat16)
draft_model.embed_tokens.requires_grad_(False)  # Use in-place operation

# Load head
head = torch.nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False, dtype=torch.bfloat16)
with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
with safe_open(os.path.join(model_path, head_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("lm_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].to(torch.bfloat16)

head.weight.data = tensor
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
    fsdp="full_shard",  # Use FSDP full_shard strategy
    fsdp_config={
        "fsdp_min_num_params": 2000,  # Wrap layers with >2000 params
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],  # Wrap Llama layers
        "fsdp_use_orig_params": True,  # Needed for gradient checkpointing
        "fsdp_cpu_ram_efficient_loading": False,  # Set to True if you have CPU memory constraints
        "fsdp_sync_module_states": True,  # Sync states across processes
        "fsdp_backward_prefetch": "backward_pre",  # Prefetch gradients during backward pass
        "fsdp_forward_prefetch": False,  # Don't prefetch in forward pass (dynamic graphs)
        "fsdp_offload_params": False,  # Set to True to offload params to CPU (saves GPU memory)
        "fsdp_sharding_strategy": "full_shard",  # Can also be "shard_grad_op" for ZeRO-2 style
    },
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
    report_to=["wandb"] if (not dist.is_initialized() or dist.get_rank() == 0) else [],  # Only log to wandb on rank 0
    log_on_each_node=False,  # Only log on main process
    logging_dir=f'{args.output_dir}/logs',  # TensorBoard log dir
)

# Handle FSDP device placement for head module
class EagleTrainerWithFSDP(EagleTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._head_device = None
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Move head to the same device as the inputs on first use
        if self._head_device is None or self._head_device != inputs["input_ids"].device:
            self._head_device = inputs["input_ids"].device
            self.head = self.head.to(device=self._head_device, dtype=torch.bfloat16)
        
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        # Move head to the same device as the inputs on first use
        if self._head_device is None or self._head_device != inputs["input_ids"].device:
            self._head_device = inputs["input_ids"].device
            self.head = self.head.to(device=self._head_device, dtype=torch.bfloat16)
        
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

trainer = EagleTrainerWithFSDP(
    model=draft_model,
    head=head,
    args=training_args,
    train_dataset=eagle_train_dataset,
    eval_dataset=eagle_test_dataset,
    data_collator=eagle_collator,
    min_lr_ratio=0.5,  # Custom lr scheduler param
)

trainer.train()

# Only push to hub and finish wandb on rank 0
if not dist.is_initialized() or dist.get_rank() == 0:
    trainer.push_to_hub(hf_repo)
    wandb.finish()  # Properly close wandb run
