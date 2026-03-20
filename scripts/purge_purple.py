import os
import numpy as np
from PIL import Image

def destroy_purple(img_path):
    print(f"Purging purple from {img_path}...")
    img = Image.open(img_path).convert("RGBA")
    data = np.array(img)
    
    r, g, b, a = data[:,:,0].astype(int), data[:,:,1].astype(int), data[:,:,2].astype(int), data[:,:,3]
    
    # Calculate how 'magenta' a pixel is.
    # High red and high blue, while having low green.
    magenta_ness = r + b - (g * 2)
    
    # If a pixel strongly leans magenta, completely neutralize it to neutral grey.
    # By forcing Red and Blue to exactly equal the Green channel, it physically cannot tint purple.
    mask = (magenta_ness > 15) & (a > 0)
    
    data[mask, 0] = data[mask, 1]
    data[mask, 2] = data[mask, 1]
    
    # Actually, if there are tiny purple pixels that just shouldn't exist (like floating specs),
    # let's just make sure they aren't almost pure background color.
    dist_to_magenta = np.sqrt((r - 255)**2 + (g - 0)**2 + (b - 255)**2)
    # If the pixel was incredibly close to magenta, just delete it entirely.
    delete_mask = (dist_to_magenta < 150) & (a > 0)
    data[delete_mask, 3] = 0
    
    # We will also make ANY remaining barely-purple pixels explicitly grey
    data[mask, 0] = data[mask, 1]
    data[mask, 2] = data[mask, 1]
    
    Image.fromarray(data).save(img_path)
    print("Done!")

destroy_purple("output/nexo/hacking_pda.png")
