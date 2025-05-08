"""
Created by Jude Wells 2025-05-08

Uses the test set visualisations of the predicted ligand
true ligand and both of these in the pocket.

Takes the outer directory which has many different systems inside
each system has an image of the voxelized predicted ligand, true ligand
and each of these in the pocket for each one we have multiple images showing
the voxels from different angles.

The script generates images which combine the following 4:
- top left: predicted ligand
- top right: true ligand
- bottom left: predicted ligand in pocket
- bottom right: true ligand in pocket

finally the script generates a gif of the different angles for the images.
"""

import os
import glob
import tarfile
from PIL import Image
import imageio
import numpy as np

def crop_by_percent(img, percent):
    """Crop a fixed percent from each side of the image."""
    w, h = img.size
    dx = int(w * percent)
    dy = int(h * percent)
    return img.crop((dx, dy, w - dx, h - dy))

def combine_images(pred_path, true_path, pred_in_pocket, true_in_pocket, output_path, combined_size=512):
    """
    Combine 4 images into a single image with 2x2 grid layout.
    Ligand-only images: crop 15% from each side.
    In-pocket images: crop 5% from each side.
    All cropped images are resized to 1/4 of the combined image size.
    """
    quarter = combined_size // 2
    # Open and crop images
    pred_img = crop_by_percent(Image.open(pred_path), 0.15)
    true_img = crop_by_percent(Image.open(true_path), 0.15)
    pred_pocket_img = crop_by_percent(Image.open(pred_in_pocket), 0.05)
    true_pocket_img = crop_by_percent(Image.open(true_in_pocket), 0.05)
    # Resize
    pred_img = pred_img.resize((quarter, quarter), Image.LANCZOS)
    true_img = true_img.resize((quarter, quarter), Image.LANCZOS)
    pred_pocket_img = pred_pocket_img.resize((quarter, quarter), Image.LANCZOS)
    true_pocket_img = true_pocket_img.resize((quarter, quarter), Image.LANCZOS)
    # Create combined image
    combined = Image.new('RGB', (combined_size, combined_size), (255, 255, 255))
    combined.paste(pred_img, (0, 0))  # top left
    combined.paste(true_img, (quarter, 0))  # top right
    combined.paste(pred_pocket_img, (0, quarter))  # bottom left
    combined.paste(true_pocket_img, (quarter, quarter))  # bottom right
    combined.save(output_path)
    return output_path

def create_gif(image_paths, output_path, duration=0.2):
    image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.png', '')))
    # add the reverse path
    image_paths.extend(image_paths[::-1])
    images = [imageio.imread(path) for path in image_paths]
    print(f"Creating GIF with {len(images)} frames, duration per frame: {duration} seconds")
    # Convert duration to milliseconds for imageio
    duration_ms = int(duration * 1000)
    imageio.mimsave(output_path, images, duration=duration_ms, loop=0)

if __name__ == "__main__":
    tar_dir = "evaluation_results/CombinedHiQBindCkptFrmPrevCombined_2025-05-06_v3_member_zero_v3/cluster_images/voxel_visualizations_all_angles_cluster"
    gif_output_dir = tar_dir.replace("voxel_visualizations_all_angles_cluster", "gifs")
    os.makedirs(gif_output_dir, exist_ok=True)
    tar_files_pattern = os.path.join(tar_dir, "*.tar.gz")
    tar_files = glob.glob(tar_files_pattern)

    for tar_file in tar_files:
        print(f"Processing {tar_file}")
        if not os.path.exists(tar_file.replace(".tar.gz", "")):
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=tar_dir)

        system_dir = os.path.join(tar_dir, os.path.basename(tar_file).replace('.tar.gz', ''))
        system_name = os.path.basename(system_dir)
        predicted_voxel_img_paths = glob.glob(os.path.join(system_dir, "predicted_ligand_angle_*.png"))
        output_dir = os.path.join(system_dir, "combined_images")
        os.makedirs(output_dir, exist_ok=True)
        combined_image_paths = []
        for pred_path in predicted_voxel_img_paths:
            true_path = pred_path.replace("predicted_ligand", "true_ligand")
            pred_in_pocket = pred_path.replace("_angle_", "_in_protein_angle_")
            true_in_pocket = true_path.replace("_angle_", "_in_protein_angle_")
            if not all([
                os.path.exists(pred_path),
                os.path.exists(true_path),
                os.path.exists(pred_in_pocket),
                os.path.exists(true_in_pocket)
            ]):
                print(f"Missing files for {system_dir}")
                continue
            angle_num = os.path.basename(pred_path).split('_')[-1].replace('.png', '')
            output_path = os.path.join(output_dir, f"combined_angle_{angle_num}.png")
            combined_path = combine_images(pred_path, true_path, pred_in_pocket, true_in_pocket, output_path, combined_size=512)
            combined_image_paths.append(combined_path)
        if combined_image_paths:
            gif_path = os.path.join(gif_output_dir, f"{system_name}.gif")
            create_gif(combined_image_paths, gif_path, duration=0.35)
            print(f"Created GIF at {gif_path}")
            # delete the combined images
            for combined_path in list(set(combined_image_paths)):
                os.remove(combined_path)
        
        # Get the case directory
        