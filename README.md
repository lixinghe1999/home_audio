# Home Audio Setup Guide

This guide outlines the steps to set up a home audio environment, including installing Miniconda, creating a Conda environment, and recording audio with a ReSpeaker device on a Linux system (e.g., Raspberry Pi).

## 1. Install Miniconda

Download and install Miniconda for Linux on an ARM64 architecture.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash ~/Miniconda3-latest-Linux-aarch64.sh
```

Follow the prompts to complete the installation.

## 2. Set Up the Environment

Create and activate a Conda environment for the home audio project, then install dependencies.

```bash
conda create --name home_audio python=3.9
conda activate home_audio
pip install -r requirements.txt
sudo apt-get install portaudio19-dev
```

Ensure the `requirements.txt` file is present in your working directory to install the necessary Python packages. The `portaudio19-dev` package is required for audio processing.

## 3. Record with ReSpeaker

Clone the ReSpeaker USB 4-mic array repository and run the recording and Direction of Arrival (DOA) scripts.

```bash
git clone https://github.com/lixinghe1999/usb_4_mic_array
python respeaker_record.py --dataset_folder recording --duration -1
sudo /home/pi/miniconda3/envs/home_audio/bin/python DOA.py
```

- The `respeaker_record.py` script records audio and saves it to the specified `recording` folder.
- The `--duration -1` flag enables continuous recording until manually stopped.
- The `DOA.py` read the hardware algorithm of DOA, which needs permission to access the hardware.