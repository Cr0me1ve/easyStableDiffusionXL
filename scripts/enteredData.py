import json

def loadFromFile(fileName: str):
    with open(f"data/{fileName}", 'r') as f:
        data = json.load(f)
    return data

def saveToFile(fileName: str, prompt: str, negativePrompt: str, width: int = 1280, height: int = 720, samples: int = 50, count: int = 1):
    d = {"prompt" : prompt,
         "negPrompt" : negativePrompt,
         "width": width,
         "height": height,
         "samples": samples,
         "count": count}
    with open(f"data/{fileName}", 'w') as f:
        json.dump(d, f)