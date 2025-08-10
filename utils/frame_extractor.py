'''
Script to extract image frames from multiple video using parallel processing and GPU.
'''
import os
import sys
import yaml # Import the YAML library
import ffmpeg
import multiprocessing
from pathlib import Path
from functools import partial

SCRIPT_PATH = Path(__file__).resolve()
UTILS_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = UTILS_DIR.parent

def load_config():
    """Loads settings from the config.yaml file."""
    config_path = PROJECT_ROOT/"config"/"config.yaml"
    if not config_path.exists():
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)

def convert_video_to_frames(video_path, config):
    """
    Converts a single video file to a sequence of image frames.
    Configuration is passed as an argument.

    Args:
        video_path (Path): The full path to the video file.
        config (dict): The loaded configuration dictionary.
    """
    print(f"[{os.getpid()}] Processing video: {video_path.name}")
    
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
    output_pattern = str(output_dir / f"f_%06d_{video_name}.{image_format}")
    
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
        print(f"[{os.getpid()}] ✅ Finished: {video_path.name}")
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
        print(f"Error: Source folder not found at '{source_path}'")
        sys.exit(1)

    output_path.mkdir(exist_ok=True)
    print(f"Configuration loaded from config.yaml")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Source folder: {source_path}")
    print(f"Output folder: {output_path}")

    # Find all .mp4 files
    video_files = list(source_path.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in '{source_path}'.")
        return

    print(f"Found {len(video_files)} videos to process.")

    # Determine the number of processes
    if isinstance(worker_processes_config, int):
        num_processes = worker_processes_config
    else: # Default to 'auto'
        num_processes = os.cpu_count()
    
    print(f"Starting conversion with {num_processes} parallel processes...")

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
        
        print("\n--- All tasks complete ---")
        print(f"Successfully converted: {len(success_files)}")
        print(f"Failed conversions:   {len(failed_files)}")

         # If there were failures, print the list of failed files
        if failed_files:
            print("\nThe following files failed to process (likely corrupted):")
            for filename in failed_files:
                print(f"  - {filename}")

if __name__ == "__main__":
    main()