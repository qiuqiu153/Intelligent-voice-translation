from model import translator as translate
import time
if __name__=="__main__":
    sentence = "My name is qiu,what about you?"
    start_time = time.perf_counter()
    translated = translate(sentence)
    end_time = time.perf_counter()
    print(f'Translated: {translated}')
    print(f'Time: {end_time - start_time:.2f}s')