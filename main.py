from SpeechRecognition.model import pipe,prepare_srt
from Translation.model import translator
from SpeechSynthesis.model import SpeechSynthesis
import librosa
import openvino as ov
import numpy as np
import soundfile
from bark import SAMPLE_RATE
import gc

def main(audio_file,output_file,low_mem=False):
    global pipe,prepare_srt,translator,SpeechSynthesis
    core=ov.Core()
    device=core.available_devices[0]
    ##读取音频文件:
    audio, sampling_rate =librosa.load(audio_file, sr=16000)
    ##将音频转换为文字
    result = pipe(audio, return_timestamps=True)

    srt_lines,_ = prepare_srt(result)
    ##将文字翻译
    translated = translator(srt_lines)[0]["translation_text"]

    if low_mem:
        del pipe
        del prepare_srt
        del translator
        gc.collect()


    ##将翻译后的文字转化为音频
    #得到音频转化对象
    generate_audio=SpeechSynthesis(device)
    #将文字转化为音频数组
    audio_array = generate_audio(translated)
    #反归一化
    audio_array = (audio_array * 32767).astype(np.int16)
    #将音频保存
    soundfile.write(output_file, audio_array, SAMPLE_RATE)

if __name__=="__main__":
    audio_file="test1.wav"
    output_file="output1.wav"
    #当你的设备只有较低的内存容量
    main(audio_file=audio_file,output_file=output_file,low_mem=True)

    #当你的设备有较高的内存容量
    # main(audio_file=audio_file,output_file=output_file)





   

    

    