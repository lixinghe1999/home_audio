# home_audio

1. install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash ~/Miniconda3-latest-Linux-aarch64.sh

2. setup environment
conda create --name home_audio python=3.9
conda activate home_audio
pip install -r requirements.txt

sudo apt-get install portaudio19-dev

3. record with respeaker
python respeaker_record.py --dataset_folder recording --duration -1
sudo /home/pi/miniconda3/envs/home_audio/bin/python DOA.py
