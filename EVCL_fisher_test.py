import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import zipfile
import json
import pyro
from pyro.nn.module import to_pyro_module_

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from accelerate import init_empty_weights
from datasets import Dataset
from huggingface_hub import login
from peft.tuners.lora import LoraLayer
from pyro.nn.module import to_pyro_module_
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from torch.amp import autocast, GradScaler
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyro.nn.module import PyroModule, PyroParam
from torch.distributions import constraints
import pyro
from torch.nn import ModuleDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "Iterative-SSR-and-EVCL-Catastrophic-Forgetting/SSR/OLD/QA_FineTuned/finetuned-weights-LoRA-EVCL"

# Load the fine-tuned tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)

# Disable caching for gradient checkpointing
model.config.use_cache = False
model.gradient_checkpointing_enable()

target_file = "Iterative-SSR-and-EVCL-Catastrophic-Forgetting/SSR/OLD/QA_FineTuned/task024_cosmosqa_answer_generation.json"

with open(target_file, 'r', encoding='utf-8-sig') as f:
    json_data = json.load(f)

instances = json_data['Instances'][0:2223]
input_texts = [str(instance['input']) for instance in instances]
output_texts = [str(instance['output'][0]) if instance['output'] else "" for instance in instances]

# Tokenize the dataset
max_seq_length = 256  # Reduce max length for memory efficiency
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length
    )
    labels = tokenizer(
        examples["output"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Create Hugging Face Dataset and tokenize
ds = Dataset.from_dict({'input': input_texts, 'output': output_texts})
tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["input", "output"])
tokenized_datasets.set_format("torch")

# Split dataset into train and eval
train_size = int(0.9 * len(tokenized_datasets))
train_dataset = tokenized_datasets.select(range(train_size))
eval_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))

# DataLoaders
batch_size = 4  # Reduced batch size for memory efficiency
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Compute Fisher Information
def compute_fisher_info(model, data_loader, num_epochs=1):
    fisher = {}
    model.train()
    scaler = GradScaler(device='cuda')

    # Initialize Fisher matrix
    for name, param in model.named_parameters():
        if 'lora' in name:
            fisher[name] = torch.zeros_like(param).to(DEVICE)

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        for i, batch in enumerate(data_loader):
            print(f"Processing batch {i + 1}/{len(data_loader)}")
            model.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            try:
                with autocast(device_type='cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
            except RuntimeError as e:
                print(f"Error in batch {i + 1}: {e}")
                break

            # Accumulate Fisher information
            for name, param in model.named_parameters():
                if 'lora' in name and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            print(f"Completed batch {i + 1}/{len(data_loader)}")

        # Normalize Fisher information after each epoch
        for name in fisher:
            fisher[name] = fisher[name] / len(data_loader)

        print(f"Completed Epoch {epoch + 1}/{num_epochs}")

    return fisher

# Function to get variational posterior means
class PyroLoraModel(PyroModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for name, module in self.model.named_modules():
            # Check for LoRA attributes (example: lora_A, lora_B)
            if hasattr(module, 'lora_A'):
                if isinstance(module.lora_A, torch.Tensor):
                    # Register tensor directly
                    module.lora_A_loc = PyroParam(
                        torch.zeros_like(module.lora_A), constraint=constraints.real
                    )
                elif isinstance(module.lora_A, (dict, ModuleDict)):
                    # Iterate through the dictionary or ModuleDict
                    for key, param in module.lora_A.items():
                        if isinstance(param, torch.Tensor):  # Ensure it's a tensor
                            module.lora_A[key + "_loc"] = PyroParam(
                                torch.zeros_like(param), constraint=constraints.real
                            )
                elif isinstance(module.lora_A, torch.nn.Module):
                    # Handle cases where lora_A is a module (e.g., Linear)
                    if hasattr(module.lora_A, 'weight'):
                        module.lora_A_loc = PyroParam(
                            torch.zeros_like(module.lora_A.weight), constraint=constraints.real
                        )

            if hasattr(module, 'lora_B'):
                if isinstance(module.lora_B, torch.Tensor):
                    # Register tensor directly
                    module.lora_B_loc = PyroParam(
                        torch.zeros_like(module.lora_B), constraint=constraints.real
                    )
                elif isinstance(module.lora_B, (dict, ModuleDict)):
                    # Iterate through the dictionary or ModuleDict
                    for key, param in module.lora_B.items():
                        if isinstance(param, torch.Tensor):  # Ensure it's a tensor
                            module.lora_B[key + "_loc"] = PyroParam(
                                torch.zeros_like(param), constraint=constraints.real
                            )
                elif isinstance(module.lora_B, torch.nn.Module):
                    # Handle cases where lora_B is a module (e.g., Linear)
                    if hasattr(module.lora_B, 'weight'):
                        module.lora_B_loc = PyroParam(
                            torch.zeros_like(module.lora_B.weight), constraint=constraints.real
                        )




pyro_model = PyroLoraModel(model)

# Verify registered parameters
print("Registered Pyro parameters:")
for name, param in pyro.get_param_store().items():
    print(name, param.shape)

# Get variational posterior means
def get_variational_posterior_means():
    posterior_means = {}
    for name, module in pyro_model.named_modules():
        if hasattr(module, 'lora_A_loc'):
            posterior_means[f"{name}.lora_A"] = pyro.param(f"{name}.lora_A_loc").detach().clone()
        if hasattr(module, 'lora_B_loc'):
            posterior_means[f"{name}.lora_B"] = pyro.param(f"{name}.lora_B_loc").detach().clone()
    return posterior_means

# Fetch posterior means
prev_posterior_means = get_variational_posterior_means()
print("Posterior Means:")
for name, mean in prev_posterior_means.items():
    print(name, mean.shape)