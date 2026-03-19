# Minecraft Texture Generator

This project automates the process of turning AI-generated (Nano Banana) images into perfect, scalable, Minecraft-ready sprite textures.

## Features
- Dynamic background detection (top-left pixel)
- Strict binary alpha enforcement (100% opaque or 100% transparent)
- Halo defringing for borderless mode
- Programmatic rarity color replacement (common, uncommon, rare, epic, legendary, relic)
- Crop and rescale using Nearest-Neighbor interpolation (16x16 to 512x512)
- Multi-skin system with nested directory preservation

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `prompts.json` with your item IDs and AI prompts.
3. Place AI-generated source images (with solid backgrounds) into the `raw/` directory.
4. Run the processor:
   ```bash
   python process_textures.py
   ```

## Workflow Summary
- **Load Manifest**: Scans `prompts.json`.
- **Extract & Enforce Alpha**: Strips backgrounds and ensures binary transparency.
- **Recolor**: Automatically handles rarity borders.
- **Export**: Saves finished `.png` sprites to `output/`.
