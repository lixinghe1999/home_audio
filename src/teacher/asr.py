from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import soundfile as sf
import os
def split_audio_timestamp(text, timestamps, audio_path, output_dir):
    """
    Split audio based on text and timestamps.
    """

    audio, sample_rate = sf.read(audio_path)   
    os.makedirs(output_dir, exist_ok=True) 
    for i, (t, ts) in enumerate(zip(text, timestamps)):
        start_time, end_time = ts
        start_sample = int(start_time /1000  * sample_rate); end_sample = int(end_time/ 1000* sample_rate)
        audio_segment = audio[start_sample:end_sample]
        output_path = f"{output_dir}/segment_{i+1}_{t}.wav"
        sf.write(output_path, audio_segment, sample_rate)

class SenseVoice_ASR():
    def __init__(self,):
        model_dir = "iic/SenseVoiceSmall"
        self.model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )
    def automatic_speech_recognition(self, audio_path):
        res = self.model.generate(
        input=f"{self.model.model_path}/example/zh.mp3",
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
        output_timestamp=True,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        timestamps = res[0]["timestamp"]
        print(text, timestamps, len(text), len(timestamps))
        return text, timestamps
    
if __name__ == "__main__":
    asr = SenseVoice_ASR()
    audio_path = "iic/SenseVoiceSmall/example/zh.mp3"
    text, timestamps = asr.automatic_speech_recognition(audio_path)
    split_audio_timestamp(text, timestamps, audio_path, "./resources/outputs")
    
