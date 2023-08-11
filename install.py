from diffusers import DiffusionPipeline
import os

try:
    os.makedirs(".cache")
    os.makedirs(".temp")
    os.makedirs("models/SD")
except:
    pass

modelName = "stabilityai/stable-diffusion-xl-base-1.0"
variants = ["fp16", "fp32"]

selectedVar = None

print("Enter variant (1 or 2):\n")
for i in range(len(variants)):
    print(f"{i+1}) {variants[i]}")

while not (selectedVar in range(len(variants))):
    selectedVar = int(input("Enter: ")) - 1

try:
    load = DiffusionPipeline.download(modelName, variant=variants[selectedVar], cache_dir=".cache")
    os.rename(str(load).replace("\\", '/'), f"models/SD/{modelName.split('/')[-1]}_{variants[selectedVar]}")
except Exception as e:
    print("Error:", e)