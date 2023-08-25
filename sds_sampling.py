import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils import io_utils, config_utils, torch_utils


from tqdm import tqdm

def main(config):

    # Env setup
    torch_utils.seed_everything(config.seed)
    io_utils.create_dir(config.output_dir)

    # Pipe setup
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    def dummy_checker(images, **kwargs): return images, [False] * len(images)
    pipe.safety_checker = dummy_checker
    pipe = pipe.to("cuda")
    # Sample setup
    prompts = config.prior_editing_prompts if config.prior else config.default_prompts
    for prompt_id, prompt in enumerate(prompts):
        print(f"[INFO] Running the process for prompt {prompt_id}: {prompt}")
        pipe.scheduler.set_timesteps(config.n_timesteps, device="cuda")
        if config.prior:
            # Construct parameters to be optimized by SDS
            # Start with a prior image
            img = io_utils.load_image(f"data/{prompt_id}.png")
            img.requires_grad = True
            latents = nn.Parameter(torch_utils.encode(img, pipe))
        else:
            # Construct parameters to be optimized by SDS
            # We can directly optimize the latent image parameters, which is an implicit way to represent the image
            latents = nn.Parameter(torch.randn(1, 4, 64, 64, device="cuda"))

        optimizer = torch.optim.Adam([latents], lr=1e-1)
        lr_lambda_wrapper = lambda step: torch_utils.lambda_lr(step, config.n_scheduler_warmup_steps, int(config.n_sampling_steps * 1.5), num_cycles=0.5)
        lr_scheduler = LambdaLR(optimizer, lr_lambda_wrapper, -1)

        # Prepare prompt embeddings
        prompt_embeds = pipe._encode_prompt(prompt=prompt, device=config.device, num_images_per_prompt=1, do_classifier_free_guidance=True)

        # DreamFusion (SDS) sampling loop
        for step in tqdm(range(config.n_sampling_steps)):
            optimizer.zero_grad()

            # Sample a random timestep  
            t = torch.randint(0, config.max_noise, [1], dtype=torch.long, device=pipe.device)

            # Distillation loop
            with torch.no_grad():
                # Sample noise
                noise = torch.randn_like(latents)
                # Add noise to the latent image parameters
                latents_noisy = pipe.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2)
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.CFG * (noise_pred_text - noise_pred_uncond)

            # Calculate the gradient
            alphas = pipe.scheduler.alphas.to(latents.device)
            w = (1 - alphas[t])
            grad = w * (noise_pred - noise)

            # Backpropagate it on latent image parameters
            latents.backward(gradient=grad, retain_graph=True)
            optimizer.step()
            lr_scheduler.step()

            if step > 0 and step % 10 == 0:
                image = torch_utils.decode(latents, pipe)
                image.save(f"{config.output_dir}/{prompt_id}_{step:03}.png")


if __name__ == "__main__":
    config = config_utils.load_config(sys.argv)
    main(config)
