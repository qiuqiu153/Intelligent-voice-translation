from datasets import load_dataset
import librosa




import numpy as np
import os
from model import pipe,prepare_srt

if __name__=="__main__":
    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # sample_long = dataset[0]
    # print(sample_long["audio"].copy())
    audio_file="ni.m4a"
    audio, sampling_rate =librosa.load(audio_file, sr=16000)
    input_={"path":audio_file,"array":audio,"sampling_rate":sampling_rate}

    result = pipe(input_, return_timestamps=True)

    srt_lines,_ = prepare_srt(result)

    print("".join(srt_lines))