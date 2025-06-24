from pathlib import Path
import numpy as np
import os
import torch
import re
import cv2 
import logging
import subprocess
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from typing import Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file, overriding system variables
# Use find_dotenv to explicitly locate the .env file if needed
dotenv_path = find_dotenv()
logging.info(f"Loading environment variables from: {dotenv_path if dotenv_path else 'Not found'}")
load_dotenv(dotenv_path=dotenv_path, override=True)

# --- Debugging: Log raw environment variable values ---
raw_device = os.getenv('DEVICE')
raw_model_name = os.getenv('MODEL_NAME')
# raw_max_frames = os.getenv('MAX_FRAMES_TO_PROCESS') # Removed
raw_start_frame = os.getenv('START_FRAME')
raw_end_frame = os.getenv('END_FRAME')
logging.debug(f"Raw DEVICE from env: '{raw_device}'")
logging.debug(f"Raw MODEL_NAME from env: '{raw_model_name}'")
# logging.debug(f"Raw MAX_FRAMES_TO_PROCESS from env: '{raw_max_frames}'") # Removed
logging.debug(f"Raw START_FRAME from env: '{raw_start_frame}'")
logging.debug(f"Raw END_FRAME from env: '{raw_end_frame}'")
# --- End Debugging ---

# Read Hugging Face Model ID if specified
hf_model_id: Optional[str] = os.getenv('HF_MODEL_ID')
if hf_model_id:
    logging.info(f"Using specific Hugging Face Model ID: {hf_model_id}")

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

# Consider making model name configurable via env var
model_name: str = os.getenv('MODEL_NAME', 'lama') # Reads MODEL_NAME from .env, defaults to 'lama' if not set
logging.info(f"Using device: {device}")
logging.info(f"Using model: {model_name}")

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

# Load additional settings from environment variables
output_filename_template: str = os.getenv('OUTPUT_FILENAME_TEMPLATE', 'S01_SH04_workoutBuddies_clean_plate.{frame}.jpeg')
image_quality: int = int(os.getenv('IMAGE_QUALITY', 95))

# Load frame range from environment variables
start_frame_str: Optional[str] = os.getenv('START_FRAME')
end_frame_str: Optional[str] = os.getenv('END_FRAME')
start_frame: int = -1
end_frame: int = -1

try:
    if start_frame_str:
        start_frame = int(start_frame_str)
    if end_frame_str:
        end_frame = int(end_frame_str)

    if start_frame > 0 and end_frame > 0 and start_frame > end_frame:
        logging.warning(f"START_FRAME ({start_frame}) is greater than END_FRAME ({end_frame}). Processing all frames.")
        start_frame = -1
        end_frame = -1
    elif start_frame > 0 and end_frame > 0:
        logging.info(f"Processing frames from {start_frame} to {end_frame} (inclusive).")
    elif start_frame > 0:
        logging.info(f"Processing frames starting from {start_frame}.")
        end_frame = float('inf') # Effectively no upper limit if only start is set
    elif end_frame > 0:
        logging.info(f"Processing frames up to {end_frame}.")
        start_frame = 0 # Effectively start from the beginning if only end is set
    else:
        logging.info("START_FRAME or END_FRAME not set or invalid. Processing all frames.")
        start_frame = -1
        end_frame = -1

except ValueError:
    logging.warning(f"Invalid value for START_FRAME ('{start_frame_str}') or END_FRAME ('{end_frame_str}'). Processing all frames.")
    start_frame = -1
    end_frame = -1

# Create output directory if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

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

def process_image_with_iopaint(img_path: Path, mask_path: Path, output_path: Path, frame_number: str) -> bool:
    """Processes a single image using IOPaint and saves the result."""
    try:
        logging.info(f"Processing image: {img_path.name} (Frame {frame_number})")
        logging.info(f"Using mask: {mask_path.name}")
        
        # Create a temporary dir for single file processing
        temp_input_dir = output_folder / "temp_input"
        temp_mask_dir = output_folder / "temp_mask"
        temp_output_dir = output_folder / "temp_output"
        
        # Create temp directories
        temp_input_dir.mkdir(exist_ok=True)
        temp_mask_dir.mkdir(exist_ok=True)
        temp_output_dir.mkdir(exist_ok=True)
        
        # Copy files to temp dirs with consistent names
        temp_img_path = temp_input_dir / f"frame_{frame_number}.jpg"
        temp_mask_path = temp_mask_dir / f"frame_{frame_number}.png"
        
        # Copy the image and mask to temp locations
        Image.open(img_path).save(temp_img_path)
        
        # Ensure mask is properly converted to binary
        mask_pil = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_pil, dtype=np.uint8)
        mask_binary = np.where(mask_np > 127, 255, 0).astype(np.uint8)
        Image.fromarray(mask_binary).save(temp_mask_path)
        
        # Determine the model identifier to use
        model_identifier = hf_model_id if hf_model_id else model_name
        logging.info(f"Using model identifier for IOPaint: {model_identifier}")

        # Build the IOPaint batch processing command
        cmd = [
            "iopaint", "run",
            f"--model={model_identifier}", # Use hf_model_id if available, else model_name
            f"--device={device}",
            f"--image={temp_input_dir}",
            f"--mask={temp_mask_dir}",
            f"--output={temp_output_dir}"
        ]
        
        # Execute IOPaint command
        logging.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"IOPaint failed with return code {result.returncode}")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            return False
        
        # Find the output file
        output_files = list(temp_output_dir.glob(f"frame_{frame_number}*"))
        if not output_files:
            logging.error(f"No output file generated for {img_path.name}")
            return False
            
        # Save to the final output path with correct naming convention
        result_img = Image.open(output_files[0])
        result_img.save(output_path, format="JPEG", quality=image_quality)
        
        # Clean up temp files
        temp_img_path.unlink(missing_ok=True)
        temp_mask_path.unlink(missing_ok=True)
        for f in output_files:
            f.unlink(missing_ok=True)
            
        logging.info(f"Saved to: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing {img_path.name}: {e}", exc_info=True)
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

    # Filter files based on frame range if specified
    if start_frame > 0 or end_frame > 0:
        filtered_files = []
        for f in all_input_files:
            frame_num_str = extract_frame_number(f.name)
            if frame_num_str:
                try:
                    frame_num = int(frame_num_str)
                    # Check if frame is within the specified range (inclusive)
                    is_in_range = True
                    if start_frame > 0 and frame_num < start_frame:
                        is_in_range = False
                    if end_frame > 0 and frame_num > end_frame:
                        is_in_range = False
                    
                    if is_in_range:
                        filtered_files.append(f)
                except ValueError:
                    logging.warning(f"Could not convert frame number '{frame_num_str}' from {f.name} to integer for range check.")
            else:
                 # Keep files where frame number couldn't be extracted if no range is specified strictly
                 # Or decide to skip them entirely if range filtering is active
                 pass # Currently skips if frame number extraction fails

        input_files = filtered_files
        logging.info(f"Filtered {len(all_input_files)} files down to {len(input_files)} based on frame range [{start_frame if start_frame > 0 else 'start'}, {end_frame if end_frame > 0 else 'end'}].")
    else:
        input_files = all_input_files
        logging.info(f"Processing all {len(input_files)} found image files.")

    for img_path in input_files:
        frame_number = extract_frame_number(img_path.name)

        if not frame_number:
            logging.warning(f"Could not extract frame number from {img_path.name}, skipping.")
            skipped_count += 1
            continue

        # Find corresponding mask using the pre-indexed map
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

        if process_image_with_iopaint(img_path, matching_mask_path, output_path, frame_number):
            processed_count += 1
        else:
            skipped_count += 1

    logging.info("Batch inpainting complete:")
    logging.info(f"- Processed: {processed_count} images")
    logging.info(f"- Skipped: {skipped_count} images")
    
    # Clean up temp directories
    temp_dirs = [
        output_folder / "temp_input",
        output_folder / "temp_mask",
        output_folder / "temp_output"
    ]
    for dir_path in temp_dirs:
        try:
            if dir_path.exists():
                for file in dir_path.iterdir():
                    file.unlink()
                dir_path.rmdir()
        except Exception as e:
            logging.warning(f"Could not clean up temp directory {dir_path}: {e}")

if __name__ == "__main__":
    main()