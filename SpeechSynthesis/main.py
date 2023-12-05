from model import generate_audio
import librosa
from bark import SAMPLE_RATE
import numpy as np
import soundfile

output_file = "output.wav"
text = "你好，我是中国人，我是潮汕人，我是胶己人，我来自中国汕头市"
audio_array = generate_audio(text)
print(audio_array)
audio_array = (audio_array * 32767).astype(np.int16)
soundfile.write(output_file, audio_array, SAMPLE_RATE)
