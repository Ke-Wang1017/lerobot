import torch
import torch.nn as nn
from copy import deepcopy
from tensordict._td import from_modules
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import save_file


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        # combine_state_for_ensemble causes graph breaks
        self.params = from_modules(*modules, as_module=True)
        with self.params[0].data.to("meta").to_module(modules[0]):
            self.module = deepcopy(modules[0])
        self._repr = str(modules[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _call(self, params, *args, **kwargs):
        with params.to_module(self.module):
            return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return torch.vmap(self._call, (0, None), randomness="different")(
            self.params, *args, **kwargs
        )

    def __repr__(self):
        return f"Vectorized {len(self)}x " + self._repr


class Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.ensemble = Ensemble([nn.Linear(10, 10), nn.Linear(10, 10)])

    def forward(self, x):
        return self.ensemble(x)
    
    def _save_pretrained(self, save_directory):
        """Custom save method to handle TensorDict properly"""
        import os
        from pathlib import Path
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        
        # Create a simplified state dict with only the tensors we want to save
        simplified_state_dict = {}
        
        # Extract tensors from the ensemble params
        for name, param in self.named_parameters():
            simplified_state_dict[name] = param
            
        # Save using safetensors
        save_file(simplified_state_dict, os.path.join(save_directory, SAFETENSORS_SINGLE_FILE))
        
        # Save config if needed
        # config_dict = self.config.to_dict()
        # with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
        #     json.dump(config_dict, f)


model = Model()
model.save_pretrained(".")