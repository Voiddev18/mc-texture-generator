import os
import numpy as np
from PIL import Image

def strict_cutout(input_path, output_path, target_size, threshold=120):
    try:
        print(f"Aggressively cutting {input_path} with threshold {threshold}...")
        img = Image.open(input_path).convert("RGBA")
        w_orig, h_orig = img.size
        # Crop hallucinatory frame
        img = img.crop((15, 15, w_orig - 15, h_orig - 15))
        data = np.array(img)
        
        r, g, b = data[:,:,0].astype(float), data[:,:,1].astype(float), data[:,:,2].astype(float)
        bg_r, bg_g, bg_b = r[0, 0], g[0, 0], b[0, 0]
        
        # Calculate color distance to background
        color_dist = np.sqrt((r - bg_r)**2 + (g - bg_g)**2 + (b - bg_b)**2)
        
        # We find what is DEFINITELY background using the threshold
        color_mask = (color_dist < threshold)
        
        # Delete pixels that are background or halo
        data[color_mask, 3] = 0
        data[color_mask, 0] = 0
        data[color_mask, 1] = 0
        data[color_mask, 2] = 0
        
        # Force 100% opacity on everything else
        data[~color_mask, 3] = 255
        
        # We WILL ALSO force desaturate the green channel in any bordering pixels just to avoid ANY green bleeding.
        # But for frag_grenade we must be careful. We'll only do it for heavily green stuff.
        # Actually, simply deleting pixels < threshold should give a totally solid cut.
        
        transparent_img = Image.fromarray(data)
        bbox = transparent_img.getbbox()
        if not bbox: return
        
        cropped_img = transparent_img.crop(bbox)
        w, h = cropped_img.size
        scale = target_size / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        # Nearest neighbor so it stays pixel art
        resized_item = cropped_img.resize((new_w, new_h), Image.Resampling.NEAREST)
        final_img = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        final_img.paste(resized_item, ((target_size - new_w) // 2, (target_size - new_h) // 2), resized_item)
        final_img.save(output_path)
        print(f"Success: {output_path}")

    except Exception as e:
        print(f"Failed {input_path} -> {output_path}: {e}")

tasks = [
    ("raw/nexo/enforcer_role.png", "output/nexo/enforcer_role.png", 64, 120),
    ("raw/nexo/frag_grenade.png", "output/nexo/frag_grenade.png", 128, 110),
    ("raw/nexo/heist_logo.png", "output/nexo/heist_logo.png", 512, 140),
    ("raw/nexo/lockpick_set.png", "output/nexo/lockpick_set.png", 128, 130),
    ("raw/nexo/thermal_drill.png", "output/nexo/thermal_drill.png", 256, 130)
]

for raw_p, out_p, size, thresh in tasks:
    strict_cutout(raw_p, out_p, size, thresh)
