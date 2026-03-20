import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def recolor_outline(data, outline_color, new_outline_color, edge_mask=None):
    data = data.copy()
    data_float = data[:, :, :3].astype(float)
    outline = np.array(outline_color, dtype=float)
    out_var = np.sum((outline - np.mean(outline))**2)
    
    if out_var < 10: return data

    r, g, b = data_float[:,:,0], data_float[:,:,1], data_float[:,:,2]
    c_mean = (r + g + b) / 3.0
    delta_r, delta_g, delta_b = r - c_mean, g - c_mean, b - c_mean
    
    c_var = delta_r**2 + delta_g**2 + delta_b**2
    c_norm = np.where(np.sqrt(c_var) < 1e-5, 1e-5, np.sqrt(c_var))
    
    delta_out = outline - np.mean(outline)
    dot = delta_r * delta_out[0] + delta_g * delta_out[1] + delta_b * delta_out[2]
    cos_theta = dot / (c_norm * np.sqrt(out_var))
    
    inv_alpha = np.clip(dot / out_var, 0, 5)
    inv_alpha[cos_theta < 0.8] = 0
    if edge_mask is not None:
        inv_alpha[~edge_mask] = 0
        
    new_out = np.array(new_outline_color, dtype=float)
    new_r = r - inv_alpha * outline[0] + inv_alpha * new_out[0]
    new_g = g - inv_alpha * outline[1] + inv_alpha * new_out[1]
    new_b = b - inv_alpha * outline[2] + inv_alpha * new_out[2]
    
    data[:,:,0] = np.clip(new_r, 0, 255).astype(np.uint8)
    data[:,:,1] = np.clip(new_g, 0, 255).astype(np.uint8)
    data[:,:,2] = np.clip(new_b, 0, 255).astype(np.uint8)
    return data

def force_process(input_path, output_path, target_size):
    try:
        img = Image.open(input_path).convert("RGBA")
        w_orig, h_orig = img.size
        img = img.crop((15, 15, w_orig - 15, h_orig - 15))
        data = np.array(img)
        
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        bg_r, bg_g, bg_b = float(data[0, 0, 0]), float(data[0, 0, 1]), float(data[0, 0, 2])
        color_dist = np.sqrt((r.astype(float) - bg_r)**2 + (g.astype(float) - bg_g)**2 + (b.astype(float) - bg_b)**2)
        
        color_mask = (color_dist < 80)
        
        bg_img_for_dilation = Image.fromarray((color_mask * 255).astype(np.uint8))
        dilated_bg = bg_img_for_dilation.filter(ImageFilter.MaxFilter(size=5))
        dilated_bg_mask = (np.array(dilated_bg) == 255)
        
        edge_mask = dilated_bg_mask & ~color_mask
        
        data[color_mask, 3] = 0
        data[color_mask, 0] = 0
        data[color_mask, 1] = 0
        data[color_mask, 2] = 0
        data[~color_mask, 3] = 255
        
        data = recolor_outline(data, (bg_r, bg_g, bg_b), (0, 0, 0), edge_mask=edge_mask)
        
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
    ("raw/nexo/smoke_grenade.png", "output/nexo/smoke_grenade.png", 128)
]

for raw_p, out_p, size in tasks:
    force_process(raw_p, out_p, size)
