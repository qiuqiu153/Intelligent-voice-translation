from pydub import AudioSegment
import numpy as np
input_path="test2.m4a"
# 读取M4A文件
audio = AudioSegment.from_file(input_path)
audio.export(out_f='./test2.wav', format='wav')  # 用于保存剪辑之后的音频文件