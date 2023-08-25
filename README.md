# Minimal implementation of the core idea of DreamFusion (2D only)

To reproduce the following results, execute `reproduce.sh` file. The code is tested using `CompVis/stable-diffusion-v1-4` from `Higgingface Diffusers v0.19.0.dev0`.

## Experiment 1: Ancestral Sampling
Prompts:
  - "A cat sitting on a chair"
  - "A blue car parked on the street"
  - "A red apple on a wooden table"
![Ancestral Sampling](./assets/a1.gif)

## Experiment 2: Score Distillation Sampling (SDS)
Prompts:
  - "A cat sitting on a chair"
  - "A blue car parked on the street"
  - "A red apple on a wooden table"
![SDS Sampling](./assets/a2.gif)

## Experiment 3: Effect of Classifier-Free Guidance Factor
From left to right: CFG={2, 6, 10, 25, 50, 100}
![CFG Effect](./assets/a3.gif)

## Experiment 4: SDS with Prior
Editing Prompts:
  - "A dog"
  - "A red car"
  - "A pineapple"
![SDS with Prior](./assets/a4.gif)
