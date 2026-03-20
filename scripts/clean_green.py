import os
import numpy as np
from PIL import Image

def clean_leftover_bg(image_path, output_path, bg_color=(0, 255, 0), threshold=100):
    try:
        img = Image.open(image_path).convert("RGBA")
        data = np.array(img)
        
        # Extract RGB
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        
        # Calculate distance to background color
        color_dist = np.sqrt((r.astype(float) - bg_color[0])**2 + 
                             (g.astype(float) - bg_color[1])**2 + 
                             (b.astype(float) - bg_color[2])**2)
                             
        # Create mask of green pixels
        green_mask = color_dist < threshold
        
        # Set those to transparent
        data[green_mask, 3] = 0
        data[green_mask, 0] = 0
        data[green_mask, 1] = 0
        data[green_mask, 2] = 0
        
        img_out = Image.fromarray(data)
        img_out.save(output_path)
        print(f"Cleaned {image_path}")
    except Exception as e:
        print(f"Failed to clean {image_path}: {e}")

images_to_clean = [
    r"output/nexo/enforcer_role.png",
    r"output/nexo/frag_grenade.png",
    r"output/nexo/heist_logo.png",
    r"output/nexo/lockpick_set.png",
    r"output/nexo/thermal_drill.png"
]

for img_path in images_to_clean:
    clean_leftover_bg(img_path, img_path)
