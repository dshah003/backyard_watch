'''
Script to review raw data using ML model. Reject images without birds/animals and keep images which have them.
'''
import os
import sys
import yaml
import shutil
import logging
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from math import ceil


# --- Path Anchoring ---
# Find the project root directory dynamically
SCRIPT_PATH = Path(__file__).resolve()
UTILS_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = UTILS_DIR.parent
# --- End of Path Anchoring ---

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


def load_config():
    """Loads settings from the config.yaml file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        logging.error(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            sys.exit(1)

def main():
    """
    Main function to sort frames using optimized batch processing.
    """
    # 1. Load Configuration
    config = load_config()['sorter_settings']
    
    model_name = config['yolo_model']
    conf_threshold = config['confidence_threshold']
    target_class_names = set(config['target_classes'])
    batch_size = config['performance']['batch_size']

    source_dir = Path(config['source_directory'])
    positive_dir = Path(config['output_directory_positive'])
    negative_dir = Path(config['output_directory_negative'])

    # 2. Setup Environment
    logging.info(f"Loading YOLO model: {model_name}...")
    model = YOLO(model_name)
    
    model_class_names = model.names
    target_class_ids = {
        k for k, v in model_class_names.items() if v in target_class_names
    }

    logging.info(f"Searching for classes: {', '.join(target_class_names)}")
    logging.info(f"Using batch size: {batch_size}")

    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    # Create class subfolders in positive_dir ahead of time
    class_dirs = {}
    for cls_id in target_class_ids:
        class_name = model_class_names[cls_id]
        class_path = positive_dir / class_name
        class_path.mkdir(parents=True, exist_ok=True)
        class_dirs[cls_id] = class_path

    image_files = list(source_dir.rglob("*.jpg"))
    if not image_files:
        logging.error(f"Error: No .jpg files found in '{source_dir}'")
        return

    logging.info(f"Found {len(image_files)} images to process. Starting...")
    
    # 3. Process Images in Batches
    positive_count = 0
    negative_count = 0

    # Calculate the total number of batches for the progress bar
    num_batches = ceil(len(image_files) / batch_size)

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        # Create the batch of image paths
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_paths = image_files[start_index:end_index]

        if not batch_paths:
            continue

        # Run inference on the entire batch at once
        results_list = model.predict(
            source=batch_paths,
            conf=conf_threshold,
            verbose=False
        )
        
        # Now, process the results for the batch
        for img_path, results in zip(batch_paths, results_list):
            found_target = False
            for box in results.boxes:
                if int(box.cls) in target_class_ids:
                    found_target = True
                    break
            
            if found_target:
                shutil.move(str(img_path), str(class_dirs[int(box.cls)] / img_path.name))
                positive_count += 1
            else:
                shutil.move(str(img_path), str(negative_dir / img_path.name))
                negative_count += 1

    # 4. Final Report
    logging.info("\n--- Sorting Complete ---")
    logging.info(f"Images with animals/birds: {positive_count}")
    logging.info(f"Images without animals/birds: {negative_count}")
    logging.info(f"Results saved in '{positive_dir.parent}'")


if __name__ == "__main__":
    main()