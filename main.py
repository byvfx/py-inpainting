from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy
from PIL import Image
import numpy as np
import os
import torch
import re
import cv2 

from dotenv import load_dotenv
import logging 
from pathlib import Path
from typing import Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Determine processing device based on .env setting and availability
requested_device: Optional[str] = os.getenv('DEVICE', '').lower() # Read from .env, default to empty string
device: str = 'cpu' # Default to CPU

if requested_device == 'cuda':
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("CUDA requested and available. Using GPU.")
    else:
        logging.warning("CUDA requested in .env file, but it is not available. Falling back to CPU.")
        device = 'cpu'
elif requested_device == 'cpu':
    device = 'cpu'
    logging.info("CPU explicitly requested in .env file. Using CPU.")
else: # Auto-detect if DEVICE is not set or invalid
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("CUDA is available (auto-detected). Using GPU.")
    else:
        device = 'cpu'
        logging.info("CUDA not available (auto-detected). Using CPU.")


# Consider making model name configurable via env var as well
model_name: str = os.getenv('MODEL_NAME', 'lama') # Reads MODEL_NAME from .env, defaults to 'lama' if not set
logging.info(f"Using device: {device}") # Log the final determined device
logging.info(f"Loading model: {model_name}")
try:
    # Ensure the model_name is supported by lama-cleaner (e.g., lama, ldm, manga, sd1.5, sd2, paint_by_example)
    model = ModelManager(name=model_name, device=device) # Pass the determined device to the model manager
except Exception as e:
    logging.error(f"Failed to load model '{model_name}' on device '{device}': {e}")
    # Add hint for common error
    if "Not supported model" in str(e):
        logging.error("Please ensure the MODEL_NAME in the .env file is one supported by lama-cleaner (e.g., lama, ldm, manga, sd1.5, sd2).")
    exit(1)


# Folder paths from environment variables using pathlib
input_folder_path: Optional[str] = os.getenv('INPUT_FOLDER')
mask_folder_path: Optional[str] = os.getenv('MASK_FOLDER')
output_folder_path: Optional[str] = os.getenv('OUTPUT_FOLDER')

if not all([input_folder_path, mask_folder_path, output_folder_path]):
    logging.error("INPUT_FOLDER, MASK_FOLDER, or OUTPUT_FOLDER environment variables not set.")
    exit(1)

input_folder: Path = Path(input_folder_path)
mask_folder: Path = Path(mask_folder_path)
output_folder: Path = Path(output_folder_path)

output_filename_template: str = os.getenv('OUTPUT_FILENAME_TEMPLATE', 'S01_SH04_workoutBuddies_clean_plate.{frame}.jpeg')
image_quality: int = int(os.getenv('IMAGE_QUALITY', 95))
# Load max frames limit from environment variable
max_frames_str: Optional[str] = os.getenv('MAX_FRAMES_TO_PROCESS')
max_frames: int = -1
if max_frames_str:
    try:
        max_frames = int(max_frames_str)
        if max_frames <= 0:
            max_frames = -1 # Treat 0 or negative as no limit
            logging.info("MAX_FRAMES_TO_PROCESS is zero or negative, processing all frames.")
        else:
            logging.info(f"MAX_FRAMES_TO_PROCESS set to {max_frames}. Processing limit applied.")
    except ValueError:
        logging.warning(f"Invalid value for MAX_FRAMES_TO_PROCESS: '{max_frames_str}'. Processing all frames.")
        max_frames = -1
else:
    logging.info("MAX_FRAMES_TO_PROCESS not set. Processing all frames.")


output_folder.mkdir(parents=True, exist_ok=True)

# Create proper Config object with correct types
# For 1920x1080 resolution:
# - HDStrategy.ORIGINAL: Processes the full image. Good starting point for quality.
# - HDStrategy.CROP: Processes only the masked region + margin if image longer edge > trigger size.
#                    Can be faster if masks are small. 1920 > 1280, so CROP would activate.
# - HDStrategy.RESIZE: Resizes image if longer edge > resize limit. 1920 < 2048, so RESIZE wouldn't activate with current limit.
#
# Using ORIGINAL as default. If performance is an issue, consider trying HDStrategy.CROP.
# Add ldm_steps, required when using the 'ldm' model.
ldm_steps_default = 50 # Default value for LDM steps
ldm_steps: int = int(os.getenv('LDM_STEPS', ldm_steps_default))
logging.info(f"Using LDM steps: {ldm_steps}")

config = Config(
    ldm_steps=ldm_steps,                     # Number of steps for LDM sampler. Required for LDM model.
    hd_strategy=HDStrategy.ORIGINAL,         # Strategy for handling high-resolution images. ORIGINAL is recommended to start.
    hd_strategy_crop_margin=128,         # Pixels padding for CROP strategy. Ignored if strategy is not CROP.
    hd_strategy_crop_trigger_size=1280,  # Min longer edge size to trigger CROP strategy. Ignored if strategy is not CROP.
    hd_strategy_resize_limit=2048,       # Max longer edge size for RESIZE strategy. Ignored if strategy is not RESIZE.
    # Add other LDM-specific parameters here if needed, e.g., ldm_sampler
)

# Function to extract frame number from filename
def extract_frame_number(filename: str) -> Optional[str]:
    """Extracts the frame number (sequence of digits) before the final extension."""
    match = re.search(r'\.(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    logging.warning(f"Could not extract frame number from filename: {filename}")
    return None

# Pre-index mask files by frame number
logging.info(f"Indexing masks in {mask_folder}...")
mask_map = {}
for mask_file in mask_folder.iterdir():
    if mask_file.is_file() and mask_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        # Attempt to find any sequence of digits in the mask filename
        match = re.search(r'(\d+)', mask_file.name)
        if match:
            frame_num = match.group(1)
            # If multiple masks contain the same number, this will overwrite.
            # Consider how to handle this if it's a possibility (e.g., log warning, store a list)
            mask_map[frame_num] = mask_file
        else:
            logging.warning(f"Could not extract frame number from mask file: {mask_file.name}")
logging.info(f"Found {len(mask_map)} masks indexed by frame number.")


def process_image(img_path: Path, mask_path: Path, output_path: Path, frame_number: str) -> bool:
    """Processes a single image using the loaded model and saves the result."""
    try:
        logging.info(f"Processing image: {img_path.name} (Frame {frame_number})")
        logging.info(f"Using mask: {mask_path.name}")

        # Load image using PIL
        image_pil = Image.open(img_path).convert("RGB")
        image = np.array(image_pil, dtype=np.uint8)

        # Load mask using PIL
        mask_pil = Image.open(mask_path).convert("L")
        mask = np.array(mask_pil, dtype=np.uint8)

        # Ensure binary mask (adjust threshold if needed)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)

        # Print shapes for debugging if necessary
        # logging.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
        # logging.debug(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

        # Process with model
        result = model(image, mask, config)

        # Handle potential float result from model
        if result.dtype != np.uint8:
            if np.issubdtype(result.dtype, np.floating):
                if np.max(result) <= 1.0 + 1e-6: # Allow for small floating point inaccuracies
                    result = (result * 255).astype(np.uint8)
                else: # Assume range is already 0-255 if max > 1
                    result = np.clip(result, 0, 255).astype(np.uint8)
            else: # Handle other potential types if necessary
                 result = result.astype(np.uint8)

        # Convert result from BGR to RGB before saving with PIL
        if len(result.shape) == 3 and result.shape[2] == 3:
             logging.debug("Converting result from BGR to RGB color space.")
             result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Save using PIL
        result_pil = Image.fromarray(result) # Assumes result is RGB
        result_pil.save(output_path, format="JPEG", quality=image_quality)
        logging.info(f"Saved to: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing {img_path.name}: {e}", exc_info=True) # Log traceback
        return False

def main():
    """Main processing loop."""
    logging.info(f"Processing images from {input_folder}")
    processed_count = 0
    skipped_count = 0

    # Sort input files numerically based on extracted frame number if possible
    all_input_files = sorted(
        [f for f in input_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']],
        key=lambda p: int(extract_frame_number(p.name) or -1) # Sort numerically, non-matching at start
    )

    # Apply the frame limit if set
    if max_frames > 0:
        input_files = all_input_files[:max_frames]
        logging.info(f"Limiting processing to the first {len(input_files)} frames based on MAX_FRAMES_TO_PROCESS={max_frames}.")
    else:
        input_files = all_input_files


    for img_path in input_files:
        frame_number = extract_frame_number(img_path.name)

        if not frame_number:
            logging.warning(f"Could not extract frame number from {img_path.name}, skipping.")
            skipped_count += 1
            continue

        # Find corresponding mask using the pre-indexed map
        # Use zfill on frame_number if mask filenames contain leading zeros consistently
        # Example: mask_map.get(frame_number.zfill(4))
        # Current implementation checks if frame_number is *part* of the key
        matching_mask_path = None
        # Try exact match first (if mask index uses padded numbers)
        padded_frame = frame_number.zfill(4) # Assuming 4-digit padding needed
        if padded_frame in mask_map:
             matching_mask_path = mask_map[padded_frame]
        elif frame_number in mask_map: # Fallback to non-padded number
             matching_mask_path = mask_map[frame_number]
        # Add more sophisticated matching if needed based on mask naming conventions

        if not matching_mask_path:
            logging.warning(f"No mask found for frame {frame_number} ({img_path.name}), skipping.")
            skipped_count += 1
            continue

        # Create new filename with desired pattern
        # Pad frame number with leading zeros
        padded_frame_output = frame_number.zfill(4) # Ensure consistent padding for output
        new_filename = output_filename_template.format(frame=padded_frame_output)
        output_path = output_folder / new_filename

        if process_image(img_path, matching_mask_path, output_path, frame_number):
            processed_count += 1
        else:
            skipped_count += 1


    logging.info("Batch inpainting complete:")
    logging.info(f"- Processed: {processed_count} images")
    logging.info(f"- Skipped: {skipped_count} images")

if __name__ == "__main__":
    main()