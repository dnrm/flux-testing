import torch
from diffusers import FluxPipeline
import uuid

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float32,
)

prompt = "a cup of coffee with a latte art heart"

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save(f"static/{uuid.uuid4()}.png")
