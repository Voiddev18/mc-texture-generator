import os
import numpy as np
from PIL import Image

def suppress_magenta(input_path, output_path):
    try:
        print(f"Applying MAGENTA spill suppression to {input_path}...")
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)
        
        r, g, b, a = data[:,:,0].astype(int), data[:,:,1].astype(int), data[:,:,2].astype(int), data[:,:,3]
        
        # A mathematical magenta spill suppressor
        # Any magenta value (high red/blue) that significantly exceeds green is suppressed back down
        # Clamp Red and Blue so they never exceed Green + a small threshold
        
        r_clamped = np.minimum(r, g + 25)
        b_clamped = np.minimum(b, g + 25)
        
        # Only apply suppression to pixels that have SOME opacity
        mask = (a > 0)
        
        data[mask, 0] = np.clip(r_clamped[mask], 0, 255).astype(np.uint8)
        data[mask, 2] = np.clip(b_clamped[mask], 0, 255).astype(np.uint8)
        
        Image.fromarray(data).save(output_path)
        print(f"Success: {output_path}")

    except Exception as e:
        print(f"Failed {input_path} -> {output_path}: {e}")

tasks = [
    "output/nexo/hacking_pda.png"
]

for p in tasks:
    suppress_magenta(p, p)
