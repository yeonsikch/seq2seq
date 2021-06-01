import torch
import torchvision

def create(config, model_params):

    if config['type'] == 'adamw':
        return torch.optim.AdamW(
            params=model_params,
            lr=config['lr']
        )
    else:
        raise AttributeError(f'not support optmizer config: {config}')
