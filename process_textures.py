import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RARITY_COLORS = {
    "common": (128, 128, 128),
    "uncommon": (0, 255, 0),
    "rare": (0, 0, 255),
    "epic": (128, 0, 128),
    "legendary": (255, 215, 0),
    "relic": (220, 20, 60)
}

def recolor_outline(data, outline_color, new_outline_color, edge_mask=None):
    data = data.copy()
    data_float = data[:, :, :3].astype(float)
    outline = np.array(outline_color, dtype=float)
    out_mean = np.mean(outline)
    delta_out = outline - out_mean
    out_var = np.sum(delta_out**2)
    out_norm = np.sqrt(out_var)
    
    if out_var < 10:
        return data
        
    r, g, b = data_float[:,:,0], data_float[:,:,1], data_float[:,:,2]
    c_mean = (r + g + b) / 3.0
    
    delta_r = r - c_mean
    delta_g = g - c_mean
    delta_b = b - c_mean
    
    c_var = delta_r**2 + delta_g**2 + delta_b**2
    c_norm = np.sqrt(c_var)
    
    # Avoid division by zero for grayscale pixels
    c_norm = np.where(c_norm < 1e-5, 1e-5, c_norm)
    
    dot = delta_r * delta_out[0] + delta_g * delta_out[1] + delta_b * delta_out[2]
    cos_theta = dot / (c_norm * out_norm)
    
    inv_alpha = np.clip(dot / out_var, 0, 5)  # Allow higher intensity
    
    # If the color hue vector is significantly diverging from our outline target, zero it out.
    # We loosened this from 0.8 to 0.4 so it aggressively eats ALL edge highlights matching the background.
    inv_alpha[cos_theta < 0.4] = 0
    
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

def process_image(input_path, output_path, target_size, color_threshold=120, rarity_color=None, base_outline_color=None, rotate_angle=0, anchor="center"):
    try:
        # Load the image
        img = Image.open(input_path).convert("RGBA")
        
        if rotate_angle != 0:
            # Auto-detect background color to fill the new corners after rotation
            bg_corner = img.getpixel((0,0))
            # BICUBIC resampling ensures the high-res raw image rotates smoothly
            img = img.rotate(rotate_angle, expand=True, fillcolor=bg_corner, resample=Image.Resampling.BICUBIC)
            
        # Aggressively crop out the outermost 15 pixels to forcefully eliminate any 
        # hallucinatory "square frames" or "box outlines" drawn by the AI image generator.
        w_orig, h_orig = img.size
        img = img.crop((15, 15, w_orig - 15, h_orig - 15))

        data = np.array(img)

        # Extract RGB channels
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

        # Auto-detect background color based on the top-left pixel
        bg_r, bg_g, bg_b = float(data[0, 0, 0]), float(data[0, 0, 1]), float(data[0, 0, 2])
        
        # Calculate distance to the auto-detected background color
        color_dist = np.sqrt((r.astype(float) - bg_r)**2 + 
                             (g.astype(float) - bg_g)**2 + 
                             (b.astype(float) - bg_b)**2)

        # Mask where distance is below threshold (meaning it's the background)
        color_mask = (color_dist < color_threshold)
        
        mask_array = (color_mask.astype(np.uint8) * 255)
        mask_img = Image.fromarray(mask_array).copy()
        ImageDraw.floodfill(mask_img, (0, 0), 128)
        
        contiguous_bg_mask = (np.array(mask_img) == 128)
        
        # --- DYNAMIC ISLAND DESTROYER ---
        # Automatically finds pixels that match the background color but were trapped inside enclosed loops (like carrying handles)
        islands_mask = color_mask & ~contiguous_bg_mask
        
        # We merge the islands into the main background mask so they are completely deleted and their edges defringed!
        master_bg_mask = contiguous_bg_mask | islands_mask
 
        # Create an edge mask using a max filter (dilation)
        # Fix lint error by explicitly using np.array
        bg_img_for_dilation = Image.fromarray((np.array(master_bg_mask, dtype=np.uint8) * 255))
        dilated_bg = bg_img_for_dilation.filter(ImageFilter.MaxFilter(size=5))
        dilated_bg_mask = (np.array(dilated_bg) == 255)
        edge_mask = dilated_bg_mask & ~master_bg_mask

        # Set contiguous background AND internal islands to entirely transparent
        data[master_bg_mask, 3] = 0
        data[master_bg_mask, 0] = 0
        data[master_bg_mask, 1] = 0
        data[master_bg_mask, 2] = 0

        # Enforce strict binary opacity for remaining pixels
        data[~master_bg_mask, 3] = 255
        
        # --- UNIVERSAL SPILL SUPPRESSOR ---
        # If the pixel survives, we strictly clamp its dominant color so it NEVER bleeds background!
        fg_mask = ~master_bg_mask
        r_sub, g_sub, b_sub = data[fg_mask, 0].astype(int), data[fg_mask, 1].astype(int), data[fg_mask, 2].astype(int)
        
        # If the background was massively magenta (Red + Blue over Green)
        if bg_r > 200 and bg_b > 200 and bg_g < 100:
            # We cap Red and Blue to slightly above Green to physically erase Magenta
            data[fg_mask, 0] = np.clip(np.minimum(r_sub, g_sub + 25), 0, 255).astype(np.uint8)
            data[fg_mask, 2] = np.clip(np.minimum(b_sub, g_sub + 25), 0, 255).astype(np.uint8)
            
        # If the background was massively green
        elif bg_g > 200 and bg_r < 100 and bg_b < 100:
            # We cap Green to physically erase Green
            max_rb = np.maximum(r_sub, b_sub)
            data[fg_mask, 1] = np.clip(np.minimum(g_sub, max_rb + 15), 0, 255).astype(np.uint8)

        # Redefine mask to just be the master bg for the rest of processing
        mask = master_bg_mask

        if rarity_color and base_outline_color:
            data = recolor_outline(data, base_outline_color, rarity_color)
        else:
            # Defringe/remove background color halo only on edges
            # When an image doesn't have a rarity outline, it still has an anti-aliased edge
            # blending into the background color (e.g. magenta).
            # By recoloring the bg color to (0,0,0) only on the edge mask, we strip the tint robustly.
            data = recolor_outline(data, (bg_r, bg_g, bg_b), (0, 0, 0), edge_mask=edge_mask)

        # Create new image from data
        transparent_img = Image.fromarray(data)

        # Find bounding box
        bbox = transparent_img.getbbox()
        if not bbox:
            print(f"Warning: {input_path} seems completely empty or fully green.")
            return

        # Crop to the bounding box of the actual item
        cropped_img = transparent_img.crop(bbox)

        # Determine the target dimension for the crop
        # We want to fit it nicely into target_size x target_size, keeping aspect ratio
        w, h = cropped_img.size
        max_dim = max(w, h)
        
        # Calculate scale factor
        scale = target_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)

        # We need to make sure dimensions are at least 1
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # Resize the item to fit within target bounding box
        # Nearest neighbor is used to preserve pixel art style
        resized_item = cropped_img.resize((new_w, new_h), Image.Resampling.NEAREST)

        # Create a blank target_size x target_size transparent image
        final_img = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

        # Paste the resized item into the target area
        if anchor == "bottom_left":
            offset_x = 0
            offset_y = target_size - new_h
        else:
            offset_x = (target_size - new_w) // 2
            offset_y = (target_size - new_h) // 2
            
        final_img.paste(resized_item, (offset_x, offset_y), resized_item)

        # Save to output
        final_img.save(output_path)
        print(f"Processed {input_path} -> {output_path} (Size: {target_size}x{target_size})")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_entry(raw_dir, output_dir, raw_filename_base, size, rarities, base_rarity, out_dir_structure, rotate_angle=0, anchor="center"):
    possible_exts = [".png", ".jpg", ".jpeg"]
    input_path = None
    for ext in possible_exts:
        temp_path = os.path.join(raw_dir, f"{raw_filename_base}{ext}")
        if os.path.exists(temp_path):
            input_path = temp_path
            break

    if input_path:
        base_outline_color = RARITY_COLORS.get(base_rarity.lower(), (128, 0, 128))
        target_out_dir = os.path.join(output_dir, out_dir_structure)
        
        if rarities and isinstance(rarities, list):
            os.makedirs(target_out_dir, exist_ok=True)
            for rarity in rarities:
                output_path = os.path.join(target_out_dir, f"{rarity}.png")
                r_color = RARITY_COLORS.get(rarity.lower())
                if r_color:
                    process_image(input_path, output_path, size, rarity_color=r_color, base_outline_color=base_outline_color, rotate_angle=rotate_angle, anchor=anchor)
        else:
            out_path = f"{target_out_dir}.png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            process_image(input_path, out_path, size, rotate_angle=rotate_angle, anchor=anchor)
    else:
        print(f"Source image for {raw_filename_base} not found in {raw_dir}/")

def main():
    prompts_file = "prompts.json"
    raw_dir = "raw"
    output_dir = "output"

    if not os.path.exists(prompts_file):
        print(f"File {prompts_file} not found.")
        return

    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    for item in prompts:
        item_id = item.get("id")
        size = item.get("size")

        if not item_id or not size:
            print(f"Skipping invalid entry: {item}")
            continue

        base_rarity = item.get("base_rarity", "epic")
        rotate_angle = item.get("rotate", 0)
        anchor = item.get("anchor", "center")
        process_entry(raw_dir, output_dir, item_id, size, item.get("rarities"), base_rarity, item_id, rotate_angle=rotate_angle, anchor=anchor)
        
        if "skins" in item:
            for skin in item["skins"]:
                skin_id = skin.get("id")
                if not skin_id: 
                    continue
                raw_name = f"{item_id}_{skin_id}"
                out_struct = os.path.join(item_id, "skins", skin_id)
                skin_base_rarity = skin.get("base_rarity", base_rarity)
                skin_rotate = skin.get("rotate", rotate_angle)
                skin_anchor = skin.get("anchor", anchor)
                process_entry(raw_dir, output_dir, raw_name, size, skin.get("rarities"), skin_base_rarity, out_struct, rotate_angle=skin_rotate, anchor=skin_anchor)

if __name__ == "__main__":
    main()
