import sounddevice as sd
import wave
import os
import datetime

def get_device_index_by_list(device_names):
    for device_name in device_names:
        idx, device_name = get_device_index_by_name(device_name)
        if device_name != 'default':
            return idx, device_name
    return 0, 'default'

def get_device_index_by_name(device_name, record=True): 
    print(f'Looking for device: {device_name}')
    devices = sd.query_devices()
    if record: # only keep the microphone, input devices
        devices = [device for device in devices if device['max_input_channels'] > 0]
    else: # only keep the speaker, output devices
        devices = [device for device in devices if device['max_output_channels'] > 0]
    
    devices = [(device['index'], device['name']) for i, device in enumerate(devices)]
    matching_devices = [(index, name) for index, name in devices if device_name.lower() in name.lower()]

    if matching_devices:
        # Sort by index priority (lower index preferred)
        matching_devices.sort(key=lambda x: x[0])
        return matching_devices[0][0], matching_devices[0][1] # Return the index of the first match
    return devices[0]


def receive_audio(dataset_folder, device, duration=5):
    '''
    receive by sounddevice and save the audio data
    '''
    # if type(device) == str:
    #     idx, device_name = get_device_index_by_name(device)
    # else:
    if isinstance(device, list):
        idx, device_name = get_device_index_by_list(device)
    else:
        idx = device
        device_name = sd.query_devices(idx)['name']
    # Set the parameters
    sd.default.device = idx
    fs = sd.query_devices(sd.default.device[1])['default_samplerate']
    channels = sd.query_devices(sd.default.device[1])['max_input_channels']
    print(f'Using device index: {idx}, device name: {device_name}, fs: {fs}, channels: {channels}')
    # default_channels = sd.query_devices(sd.default.device[1])['max_input_channels']
    # default_sample_rate = sd.query_devices(sd.default.device[1])['default_samplerate']
    # if default_sample_rate != fs:
    #     fs = default_sample_rate
    #     channels = default_channels

    datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    filename = os.path.join(dataset_folder, f'{datetime_str}.wav')
    # Record the audio
    print('Recording audio start...')

    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()
    print('Recording audio done ...')
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(channels) 
    waveFile.setsampwidth(2)
    waveFile.setframerate(fs)
    waveFile.writeframes(myrecording)
    waveFile.close()
    print(f'Audio saved at {filename} ...')

import soundfile as sf
import queue
import sys

# Queue to hold audio data
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    """Callback function to put audio data into the queue."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def continuous_record(dataset_folder, device):
    """Record audio continuously and save to 1-minute FLAC segments."""
    # Audio recording parameters
    BLOCKSIZE = 1024  # Number of frames per callback
    DTYPE = 'float32'  # Data type for audio samples
    SEGMENT_DURATION = 60  # Duration of each segment in seconds

    try:
        # Get device index and name
        if isinstance(device, list):
            idx, device_name = get_device_index_by_list(device)
        else:
            idx = device
            device_name = sd.query_devices(idx)['name']

        # Set the parameters
        sd.default.device = idx
        fs = sd.query_devices(sd.default.device[1])['default_samplerate']
        channels = sd.query_devices(sd.default.device[1])['max_input_channels']
        print(f'Using device index: {idx}, device name: {device_name}, fs: {fs}, channels: {channels}')

        # Calculate samples per segment
        samples_per_segment = int(fs * SEGMENT_DURATION)
        samples_written = 0
        file = None

        def open_new_file():
            """Open a new FLAC file with a timestamped filename."""
            datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            filename = os.path.join(dataset_folder, f'{datetime_str}.flac')
            return sf.SoundFile(filename, mode='x', samplerate=int(fs), 
                               channels=int(channels), subtype='PCM_16')

        # Start the audio stream
        with sd.InputStream(samplerate=int(fs), channels=int(channels), 
                           dtype=DTYPE, blocksize=BLOCKSIZE, 
                           callback=callback):
            print("Recording... Press Ctrl+C to stop.")
            while True:
                # Open a new file if none exists
                if file is None:
                    file = open_new_file()
                    print(f"Started new segment: {file.name}")

                # Get audio data from queue
                data = audio_queue.get()
                num_samples = data.shape[0]
                samples_written += num_samples

                # Write data to file
                file.write(data)

                # Check if segment is complete
                if samples_written >= samples_per_segment:
                    file.close()
                    print(f"Completed segment: {file.name}")
                    file = None
                    samples_written = 0

    except KeyboardInterrupt:
        print("\nRecording stopped.")
        if file is not None:
            file.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        if file is not None:
            file.close()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Record audio from a specified device.')
    parser.add_argument('--dataset_folder', type=str, default='recording', )
    parser.add_argument('--device', type=int, default=0,)
    parser.add_argument('--duration', type=int, default=5, help='Duration of the recording in seconds.')

    args = parser.parse_args()
    if args.duration <= 0:
        print("Duration must be greater than 0. Recording continuously...")
        continuous_record(args.dataset_folder, args.device)
    else:
        receive_audio(args.dataset_folder, args.device, args.duration)