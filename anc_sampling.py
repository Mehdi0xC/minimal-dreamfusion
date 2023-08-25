import sys
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from utils import io_utils, config_utils, torch_utils
from tqdm import tqdm


def main(config):
    # Environment setup
    seed = 999
    torch_utils.seed_everything(seed)
    output_dir = config.output_dir
    io_utils.create_dir(output_dir)

    # Pipe setup
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    ).to(config.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    def dummy_checker(images, **kwargs):
        return images, [False] * len(images)

    pipe.safety_checker = dummy_checker

    # Inference setup
    gen = torch.Generator(device="cuda").manual_seed(seed)
    model = pipe.unet
    num_inference_steps = 50

    # Prompt iteration loop
    for prompt_id, prompt in enumerate(config.default_prompts):
        print(f"[INFO] Running the process for prompt {prompt_id}: {prompt}")
        pipe.scheduler.set_timesteps(num_inference_steps, device="cuda")
        prompt_embeds = pipe._encode_prompt(
            prompt=prompt, device=config.device, num_images_per_prompt=1, do_classifier_free_guidance=True
        )
        latents = torch.randn((1, 4, 64, 64), generator=gen, device="cuda")

        # Inference loop
        with torch.autocast("cuda"), torch.no_grad():
            extra_step_kwargs = {}
            extra_step_kwargs["generator"] = gen
            extra_step_kwargs["eta"] = 1.
            # Denoising loop
            for i, timestep in tqdm(enumerate(pipe.scheduler.timesteps), desc="Timesteps"):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
                # Predict noise
                noise_pred = model(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.CFG * (noise_pred_text - noise_pred_uncond)
                latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

                # Save image
                image = torch_utils.decode(latents, pipe)
                image.save(f"{output_dir}/{prompt_id}_{i:02}.png")


if __name__ == "__main__":
    config = config_utils.load_config(sys.argv)
    main(config)
