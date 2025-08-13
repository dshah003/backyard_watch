'''
Script to extract image frames from multiple video using parallel processing and GPU.
'''
import os
import sys
import yaml # Import the YAML library
import ffmpeg
import logging
import multiprocessing
from pathlib import Path
from functools import partial

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

def convert_video_to_frames(video_path, config):
    """
    Converts a single video file to a sequence of image frames.
    Configuration is passed as an argument.

    Args:
        video_path (Path): The full path to the video file.
        config (dict): The loaded configuration dictionary.
    """
    logging.info(f"[{os.getpid()}] Processing video: {video_path.name}")
    
    # Get settings from the config dictionary
    frame_output_folder = config['paths']['frame_output']
    image_format = config['settings']['image_format']
    image_quality = config['settings']['image_quality']
    gpu_decoder = config['settings']['gpu_decoder']
    
    # Create a directory for this video's frames
    video_name = video_path.stem
    # output_dir = Path(frame_output_folder) / video_name
    output_dir = Path(frame_output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file pattern
    output_pattern = str(output_dir / f"f_{video_name}_%06d.{image_format}")
    
    # Set output options
    output_options = {
        'qscale:v': image_quality,
        'pix_fmt': 'yuvj420p' # Standard pixel format for JPG
    }
    # PNGs do not need quality settings
    if image_format == 'png':
        del output_options['qscale:v']

    try:
        # Build and run the ffmpeg command
        (
            ffmpeg
            .input(str(video_path), hwaccel='cuda', vcodec=gpu_decoder)
            .output(output_pattern, **output_options)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logging.info(f"[{os.getpid()}] ✅ Finished: {video_path.name}")
        return True, video_path.name, None
    except ffmpeg.Error as e:
        error_message = f"[{os.getpid()}] Error processing {video_path.name} Retrying with CPU decoder.:\n{e.stderr.decode()}"
        print(error_message, file=sys.stderr)
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(output_pattern, **output_options)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logging.info(f"[{os.getpid()}] ✅ Finished: {video_path.name}")
            return True, video_path.name, None
        except ffmpeg.Error as e:
            error_message = f"[{os.getpid()}] Error processing {video_path.name} Retrying with Tonemap:\n{e.stderr.decode()}"
            try:
                (
                    ffmpeg
                    .input(str(video_path))
                    .filter('tonemap', tonemap='hable')
                    .output(output_pattern, **output_options)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                logging.info(f"[{os.getpid()}] ✅ Finished: {video_path.name}")
                return True, video_path.name, None
            except ffmpeg.Error as e:
                error_message = f"[{os.getpid()}] ❌ Error processing {video_path.name}:\n{e.stderr.decode()}"
                print(error_message, file=sys.stderr)
                return False, video_path.name, error_message



def main():
    """
    Main function to load config, find videos, and distribute tasks.
    """
    # Load settings from config.yaml
    config = load_config()

    # Get paths and settings from the config
    source_folder = config['paths']['video_source']
    output_folder = config['paths']['frame_output']
    worker_processes_config = config['performance']['worker_processes']

    source_path = Path(source_folder).resolve()
    output_path = Path(output_folder).resolve()

    if not source_path.is_dir():
        logging.error(f"Error: Source folder not found at '{source_path}'")
        sys.exit(1)

    output_path.mkdir(exist_ok=True)
    logging.info(f"Configuration loaded from config.yaml")
    logging.info(f"Project Root: {PROJECT_ROOT}")
    logging.info(f"Source folder: {source_path}")
    logging.info(f"Output folder: {output_path}")

    # Find all .mp4 files
    video_files = list(source_path.glob("*.mp4"))
    if not video_files:
        logging.error(f"No .mp4 files found in '{source_path}'.")
        return

    logging.info(f"Found {len(video_files)} videos to process.")

    # Determine the number of processes
    if isinstance(worker_processes_config, int):
        num_processes = worker_processes_config
    else: # Default to 'auto'
        num_processes = os.cpu_count()
    
    logging.info(f"Starting conversion with {num_processes} parallel processes...")

    # Create a partial function with the 'config' argument already filled in
    worker_func = partial(convert_video_to_frames, config=config)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(worker_func, video_files)
        
        # success_count = 0
        # failure_count = 0
        # Keep track of successful and failed files
        success_files = []
        failed_files = []
        
        for success, video_name, error_message in results:
            if success:
                success_files.append(video_name)
            else:
                failed_files.append(video_name)
        
        logging.info("\n--- All tasks complete ---")
        logging.info(f"Successfully converted: {len(success_files)}")
        logging.info(f"Failed conversions:   {len(failed_files)}")

         # If there were failures, print the list of failed files
        if failed_files:
            logging.error("\nThe following files failed to process (likely corrupted):")
            for filename in failed_files:
                logging.error(f"  - {filename}")

if __name__ == "__main__":
    main()