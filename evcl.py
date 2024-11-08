import torch
import pyro
import tyxe
import functools
import copy
import torch.nn.functional as F
import pyro.distributions as dist
import pyro.optim
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from transformers import LlamaForCausalLM, LlamaTokenizer
from model.mle_prior import MLEPrior
from typing import Optional, List
from coreset import update_coreset

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn.functional as F

def compute_fisher_info_llm(bnn, prev_fisher_info, data_loader, n_samples=5000, ewc_gamma=1.0):
    est_fisher_info = {}
    
    # Initialize Fisher info dictionary only for LoRA-modified Q, K, V weights across layers
    for name, param in bnn.named_parameters():
        if (
            "q_proj" in name or "k_proj" in name or "v_proj" in name
        ) and any(lora in name for lora in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]):
            est_fisher_info[name] = param.detach().clone().zero_()
    
    old_training_state = bnn.training
    bnn.eval()
    
    for index, (x, y) in enumerate(data_loader):
        if n_samples is not None and index >= n_samples:
            break
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        with torch.no_grad():
            outputs = bnn(x)  # Model outputs logits for each token in sequence
        
        for token_idx in range(outputs.shape[1]):
            target_token = y[:, token_idx]  # True token at this position
            output = outputs[:, token_idx, :]  # Logits for each token in vocab
            output.requires_grad = True
            token_prob = F.softmax(output, dim=1)
            
            # Compute negative log likelihood loss for the target token
            nll = F.cross_entropy(output, target_token)
            bnn.zero_grad()
            nll.backward(retain_graph=True if (token_idx + 1) < outputs.shape[1] else False)
            
            # Accumulate Fisher Information only for LoRA-modified Q, K, V parameters
            for name, param in bnn.named_parameters():
                if param.grad is not None and (
                    "q_proj" in name or "k_proj" in name or "v_proj" in name
                ) and any(lora in name for lora in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]):
                    est_fisher_info[name] += (token_prob[:, target_token] * (param.grad.detach() ** 2)).sum()
    
    # Normalize estimated Fisher Information by the number of samples
    est_fisher_info = {n: p / (index + 1) for n, p in est_fisher_info.items()}
    
    # Incorporate previous Fisher Information, if provided
    if prev_fisher_info is not None:
        for name in est_fisher_info:
            if name in prev_fisher_info:
                existing_values = prev_fisher_info[name]
                est_fisher_info[name] += ewc_gamma * existing_values

    bnn.train(old_training_state)
    return est_fisher_info



class VariationalBNNWithEWC(tyxe.VariationalBNN):
    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, device=None, ewc_lambda=0.0, fisher_info=None, prev_params=None):
        old_training_state = self.net.training
        self.net.train(True)
        
        # Choose ELBO loss type
        loss = TraceMeanField_ELBO(num_particles) if closed_form_kl else Trace_ELBO(num_particles)
        svi = SVI(self.model, self.guide, optim, loss=loss)
        
        def _as_tuple(x):
            if isinstance(x, (list, tuple)):
                return x
            return x,
        
        def _to(x, device):
            return map(lambda t: t.to(device) if device is not None else t, _as_tuple(x))
        
        for i in range(num_epochs):
            total_loss = 0.0
            num_batch = 1
            
            for num_batch, (input_data, target_data) in enumerate(iter(data_loader), 1):
                # Move data to the specified device
                input_data, target_data = tuple(_to(input_data, device)), tuple(_to(target_data, device))[0]

                # Calculate ELBO loss for each token
                elbo = svi.step(input_data, target_data)
                
                # EWC Regularization for LoRA-modified Q, K, V parameters only
                if ewc_lambda > 0 and fisher_info is not None:
                    ewc_loss = 0.0
                    for name, param in self.named_parameters():
                        # Apply EWC only to LoRA-specific parameters within Q, K, V matrices
                        if (
                            name in fisher_info
                            and any(proj in name for proj in ["q_proj", "k_proj", "v_proj"])
                            and any(lora in name for lora in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"])
                        ):
                            ewc_loss += (fisher_info[name] * (param - prev_params[name]) ** 2).sum()
                    
                    # Scale EWC loss
                    ewc_loss = (1.0 / 2) * ewc_loss
                    total_loss += elbo + (ewc_lambda * ewc_loss)
                else:
                    total_loss += elbo
            
            # Callback for monitoring progress (optional)
            if callback is not None and callback(self, i, total_loss / num_batch):
                break
        
        # Restore model’s initial training state
        self.net.train(old_training_state)
        return total_loss / num_batch

def update_variational_approx(bnn, train_loader, curr_coreset, num_epochs, callback, ewc_lambda, fisher_info=None, prev_params=None, finetune_coreset=False):
    # Select data for updating the variational approximation
    if not finetune_coreset:
        # Use non-coreset data (remaining data after excluding the coreset)
        non_coreset_data = list(set(train_loader.dataset) - set(curr_coreset))
        data_loader = torch.utils.data.DataLoader(non_coreset_data, batch_size=train_loader.batch_size, shuffle=True)
    else:
        # Use only the coreset data
        data_loader = torch.utils.data.DataLoader(curr_coreset, batch_size=train_loader.batch_size, shuffle=True)
    
    optim = pyro.optim.Adam({"lr": 1e-3})
    
    # Apply local reparameterization for stable training
    with tyxe.poutine.local_reparameterization():
        # Ensure we pass LoRA-specific Q, K, V parameters only in EWC
        bnn.fit(
            data_loader, optim, num_epochs, 
            device=DEVICE, callback=callback, 
            ewc_lambda=ewc_lambda, 
            fisher_info={k: v for k, v in (fisher_info or {}).items() if "q_proj" in k or "k_proj" in k or "v_proj" in k and any(lora in k for lora in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"])},  # LoRA-specific Fisher Info for Q, K, V
            prev_params={k: v for k, v in (prev_params or {}).items() if "q_proj" in k or "k_proj" in k or "v_proj" in k and any(lora in k for lora in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"])}   # LoRA-specific previous params for Q, K, V
        )


def run_evcl(
    num_tasks: int = 5,
    num_epochs: int = 10,
    experiment_name: str = 'test',
    task_config: str = '',
    batch_size: int = 8,
    coreset_size: int = 0,
    coreset_method: str = 'random',
    finetune_method: Optional[str] = None,
    model_suffix: Optional[str] = None,
    ewc_lambda: float = 100.0,
    ewc_gamma: float = 1.0,
):
    # Load LLaMA and tokenizer
    model = LlamaForCausalLM.from_pretrained("LLaMA_model_path")  # Specify your LLaMA model path
    tokenizer = LlamaTokenizer.from_pretrained("LLaMA_tokenizer_path")  # Specify your tokenizer path
    model.to(DEVICE)
    model.eval()  # Freeze all parameters except LoRA layers

    # Initialize LoRA layers (example code assumes LoRA setup elsewhere in the code)
    # Here, we assume LoRA is already applied to the LLaMA model
    # Only LoRA parameters will be fine-tuned

    # Load datasets and create data loaders
    train_loaders, test_loaders = fetch_nlp_datasets(tokenizer, batch_size, num_tasks)  # Custom function to handle NLP datasets

    # Set up variational BNN with EWC for LLaMA
    prior = MLEPrior(model)  # Initialize priors with model weights
    obs = tyxe.likelihoods.Categorical(-1)  # Suitable likelihood for NLP generative tasks
    guide = functools.partial(
        tyxe.guides.AutoNormal,
        init_scale=1e-4,
        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(model, prefix="net")
    )
    
    # Initialize Bayesian LLaMA model with LoRA-specific EWC
    bnn = VariationalBNNWithEWC(model, prior, obs, guide)
    
    prev_coreset = []
    prev_fisher_info = None
    prev_params = None

    for task_index, train_loader in enumerate(train_loaders, 1):
        print(f"Training on Task {task_index}...")

        # Update coreset
        if coreset_size > 0:
            curr_coreset = update_coreset(prev_coreset, train_loader, coreset_size, coreset_method)
        else:
            curr_coreset = []

        # Training loop for current task with variational inference and EWC
        def callback(epoch, step, loss):
            print(f"Epoch {epoch}, Step {step}, Loss: {loss}")
        
        # Fine-tune with variational inference and EWC on LoRA parameters only
        update_variational_approx(
            bnn, train_loader, curr_coreset, num_epochs, callback, ewc_lambda, fisher_info=prev_fisher_info, prev_params=prev_params
        )

        # Calculate Fisher Information Matrix for the LoRA-specific Q, K, V parameters only
        fisher_info = compute_fisher_info_llm(
            bnn, prev_fisher_info, train_loader, n_samples=5000, ewc_gamma=ewc_gamma
        )

        # Store previous LoRA parameters and Fisher Information for Q, K, V only
        prev_params = {
            name: param.detach().clone()
            for name, param in bnn.named_parameters()
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]) and "lora" in name
        }
        prev_fisher_info = {
            name: fisher_info[name]
            for name in fisher_info
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]) and "lora" in name
        }

        # Evaluate on all tasks up to current
        for j, test_loader in enumerate(test_loaders[:task_index], 1):
            print(f"Evaluating Task {j}...")
            correct, total = 0, 0
            for batch in test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                    predictions = logits.argmax(dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.numel()
            accuracy = correct / total
            print(f"Task {j} Accuracy: {accuracy:.4f}")

    print("Training completed.")

