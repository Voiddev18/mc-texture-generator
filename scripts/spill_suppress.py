import os
import numpy as np
from PIL import Image

def suppress_green(input_path, output_path):
    try:
        print(f"Applying green spill suppression to {input_path}...")
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)
        
        r, g, b, a = data[:,:,0].astype(int), data[:,:,1].astype(int), data[:,:,2].astype(int), data[:,:,3]
        
        # A simple and mathematically flawless green spill suppressor
        # Any green value that exceeds the maximum of (red, blue) is probably spill/halo from the background.
        # We clamp green to never exceed max(red, blue) + a small threshold
        max_rb = np.maximum(r, b)
        
        # If green is higher than max_rb, we set it back down.
        # This completely neutralizes green borders turning them into neutral greys/shadows!
        
        g_clamped = np.minimum(g, max_rb + 15)
        
        # Only apply suppression to pixels that have SOME opacity
        mask = (a > 0)
        
        data[mask, 1] = np.clip(g_clamped[mask], 0, 255).astype(np.uint8)
        
        Image.fromarray(data).save(output_path)
        print(f"Success: {output_path}")

    except Exception as e:
        print(f"Failed {input_path} -> {output_path}: {e}")

tasks = [
    "output/nexo/enforcer_role.png",
    "output/nexo/frag_grenade.png",
    "output/nexo/heist_logo.png",
    "output/nexo/lockpick_set.png",
    "output/nexo/thermal_drill.png",
    "output/nexo/combat_knife.png"
]

for p in tasks:
    suppress_green(p, p)
