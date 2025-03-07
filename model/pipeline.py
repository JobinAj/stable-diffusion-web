from diffusers import DiffusionPipeline

def generate_image(prompt):
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    image = pipe(prompt).images[0]
    return image
