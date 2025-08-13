# backyard_watch
AI based detection system to detect birds and animals in the backyard using feed from security camera


## Initial Setup  

Ran the following commands to setup venv and install some basic packages. This assumes you already have nvidia drivers, Cuda, and Cudnn setup and installed.

```sh 
sudo apt update 
sudo apt install python3-venv 
cd backyard_watch
python3 -m venv vision 
source vision/bin/activate 
pip install --upgrade pip
pip install jupyterlab 
pip install scikit-learn pandas numpy matplotlib 

sudo apt update
sudo apt install ffmpeg

# check if your ffmpeg build supports the NVIDIA cuvid decoder.
ffmpeg -hide_banner -decoders | grep cuvid

pip install ffmpeg-python
```

## Usage

```python3 frame_extractor.py```\

## Useful commands

Sort files by name and used to shortlist images. (press 1 to move the file to false_negatives dir)
```sh
feh --scale-down -S filename --action1 'mv %F false_negatives/'
feh --scale-down --action1 'echo^CF >> false_negatives.txt'
```

Delete files from folder A if they exist in folder B
```sh 
find folderB -maxdepth 1 -type f -name "*.jpg" -printf "%f\n" | xargs -I {} rm -v "folderA/{}"

find sorted_frames/no_animal/ -maxdepth 1 -type f -name "*.jpg" -printf "%f\n" | xargs -I {} rm -v "extracted_frames/{}"
```