import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lambda_lr(current_step, num_warmup_steps, num_training_steps, num_cycles=0.5):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_val = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265) * num_cycles * 2.0 * progress))
    return max(0.0, cosine_val.item())

def encode(img, pipe):
    img = 2 * img - 1
    posterior = pipe.vae.encode(img).latent_dist
    latents = posterior.sample() * pipe.vae.config.scaling_factor
    return latents

def decode(latents, pipe):
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        return image[0]