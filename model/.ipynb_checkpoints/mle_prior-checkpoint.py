import torch
import pyro.distributions as dist
from tyxe.priors import Prior
from utils.util import DEVICE   

class MLEPrior(Prior):
    def __init__(self, mle_net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mle_params = {}

        # Traverse all parameters, targeting LoRA-modified Q, K, V weights in all attention layers
        for name, param in mle_net.named_parameters():
            # Look for q_proj, k_proj, and v_proj weights with "lora" in their name to identify LoRA-modified Q, K, V matrices
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]) and "lora" in name:
                self.mle_params[name] = param.detach().to(DEVICE)

        def expose_fn(module, name):
            # Expose only parameters that are in the self.mle_params dictionary
            return name in self.mle_params

        self.expose_fn = expose_fn

    def prior_dist(self, name, module, param):
        # Apply MLE-based Normal prior centered on the pre-trained (detached) values for LoRA-modified Q, K, V parameters
        if name in self.mle_params:
            mle_param = self.mle_params[name]
            return dist.Normal(mle_param, torch.tensor(1.0, device=DEVICE))  # You can adjust the variance if needed
        else:
            # Fallback for parameters without specified priors
            return super().prior_dist(name, module, param)
