import os
import numpy as np
from PIL import Image

def strict_cutout_magenta(input_path, output_path, target_size, threshold=80):
    try:
        print(f"Magenta Cut {input_path} (thresh={threshold})...")
        img = Image.open(input_path).convert("RGBA")
        w_orig, h_orig = img.size
        # Crop hallucinatory frame
        img = img.crop((15, 15, w_orig - 15, h_orig - 15))
        data = np.array(img)
        
        r, g, b = data[:,:,0].astype(float), data[:,:,1].astype(float), data[:,:,2].astype(float)
        # Assuming background is magenta (255, 0, 255)
        # Calculate color distance to background
        color_dist = np.sqrt((r - 255)**2 + (g - 0)**2 + (b - 255)**2)
        
        color_mask = (color_dist < threshold)
        
        data[color_mask, 3] = 0
        data[color_mask, 0] = 0
        data[color_mask, 1] = 0
        data[color_mask, 2] = 0
        
        # Opaque
        data[~color_mask, 3] = 255
        
        # No spill suppression since magenta is easier to cut from military items
        
        transparent_img = Image.fromarray(data)
        bbox = transparent_img.getbbox()
        if not bbox: return
        
        cropped_img = transparent_img.crop(bbox)
        w, h = cropped_img.size
        scale = target_size / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        resized_item = cropped_img.resize((new_w, new_h), Image.Resampling.NEAREST)
        final_img = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        final_img.paste(resized_item, ((target_size - new_w) // 2, (target_size - new_h) // 2), resized_item)
        final_img.save(output_path)
        print(f"Success: {output_path}")

    except Exception as e:
        print(f"Failed {input_path} -> {output_path}: {e}")

tasks = [
    ("raw/nexo/hacking_pda.png", "output/nexo/hacking_pda.png", 256, 120)
]

for raw_p, out_p, size, thresh in tasks:
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    strict_cutout_magenta(raw_p, out_p, size, thresh)
