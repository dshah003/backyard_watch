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

```python3 frame_extractor.py```