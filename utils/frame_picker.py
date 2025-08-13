import os
import shutil
import yaml
import sys
import logging
from pathlib import Path
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
UTILS_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = UTILS_DIR.parent

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def load_config():
    """Loads settings from the config.yaml file."""
    config_path = PROJECT_ROOT/"config"/"config.yaml"
    if not config_path.exists():
        logging.error(f"Configuration file not found at '{config_path}'")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            sys.exit(1)


def select_and_move_images(source_dir, dest_dir, frames_to_skip):
    """
    Selects images from a source directory, skipping a specified number of frames,
    and moves them to a destination directory.

    Args:
        source_dir (str): The path to the directory containing the image frames.
        dest_dir (str): The path to the destination directory for the selected images.
        frames_to_skip (int): The number of frames to skip between each selected image.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    total_count = 0

    all_frames = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])

    # Counter for the number of frames to skip.
    skip_counter = 0

    for i, frame_name in enumerate(tqdm(all_frames, desc="Selecting and moving images")):
        if skip_counter == 0:
            # Move the frame if the skip counter is at 0.
            source_path = os.path.join(source_dir, frame_name)
            dest_path = os.path.join(dest_dir, frame_name)
            shutil.move(source_path, dest_path)
            # Reset the skip counter after moving a frame.
            skip_counter = frames_to_skip
            total_count += 1
        else:
            # Decrement the skip counter if we are skipping frames.
            skip_counter -= 1
    logging.info(f"Moved {total_count} images")

def main():
    config = load_config()

    source_folder = config["shortlister"]["source_directory"]
    destination_folder = config["shortlister"]["shortlisted_data"]
    frames_to_skip = config["shortlister"]["frames_to_skip"]

    select_and_move_images(source_folder, destination_folder, frames_to_skip)
    logging.info("Image selection and movement complete.")



if __name__ == "__main__":
    main()