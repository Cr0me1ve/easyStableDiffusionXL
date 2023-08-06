import torch
from diffusers import DiffusionPipeline

def genSingle(pipeline: DiffusionPipeline, prompt: str | None = None, width: int = 512, height: int = 512, samples: int = 50, negativePrompt: str | None = None):
    image = pipeline(prompt=prompt, negative_prompt=negativePrompt, width=width, height=height, num_inference_steps=samples).images[0]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image

def genMultiple(pipeline: DiffusionPipeline, prompt: str | None = None, negativePrompt: str | None = None, width: int = 512, height: int = 512, samples: int = 50, count: int = 10):
    images = []
    for i in range(count):
        images.append(genSingle(pipeline, prompt, width, height, samples, negativePrompt))
    return images