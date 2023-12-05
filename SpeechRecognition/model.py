from transformers import pipeline
from pathlib import Path
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoModelForSpeechSeq2Seq
import gc

import os

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pathlib import Path
module_path = Path(os.path.split(__file__)[0])

distil_model_id = "distil-whisper/distil-large-v2"
distil_model_path =module_path/ Path(distil_model_id.split("/")[-1])
quantized_distil_model_path = Path(f"{distil_model_path}_quantized")
processor = AutoProcessor.from_pretrained(distil_model_id) 

from pathlib import Path
from tqdm import tqdm
from itertools import islice
from typing import List, Any
from openvino import Tensor
import shutil
import nncf
import gc
import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from datasets import load_dataset


def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features

class InferRequestWrapper:
    def __init__(self, request, data_cache: List):
        self.request = request
        self.data_cache = data_cache

    def __call__(self, *args, **kwargs):
        self.data_cache.append(*args)
        return self.request(*args, *kwargs)

    def infer(self, inputs: Any = None, shared_memory: bool = False):
        self.data_cache.append(inputs)
        return self.request.infer(inputs, shared_memory)

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
        shared_memory: bool = False,
    ):
        self.data_cache.append(inputs)
        self.request.infer(inputs, shared_memory)

    def wait(self):
        pass

    def get_tensor(self, name: str):
        return Tensor(self.request.results[name])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)


def collect_calibration_dataset(ov_model, calibration_dataset_size):
    # Overwrite model request properties, saving the original ones for restoring later
    original_encoder_request = ov_model.encoder.request
    original_decoder_with_past_request = ov_model.decoder_with_past.request
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(original_encoder_request, encoder_calibration_data)
    ov_model.decoder_with_past.request = InferRequestWrapper(original_decoder_with_past_request,
                                                             decoder_calibration_data)

    calibration_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data",
                       total=calibration_dataset_size):
        input_features = extract_input_features(sample)
        ov_model.generate(input_features)

    ov_model.encoder.request = original_encoder_request
    ov_model.decoder_with_past.request = original_decoder_with_past_request

    return encoder_calibration_data, decoder_calibration_data
def quantize(ov_model, calibration_dataset_size=10):
    
    if not quantized_distil_model_path.exists():
        print(ov_model)
        encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(
            ov_model, calibration_dataset_size
        )
        print("Quantizing encoder")
        quantized_encoder = nncf.quantize(
            ov_model.encoder.model,
            nncf.Dataset(encoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(encoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
        )
        ov.save_model(quantized_encoder, quantized_distil_model_path / "openvino_encoder_model.xml")
        del quantized_encoder
        del encoder_calibration_data
        gc.collect()

        print("Quantizing decoder with past")
        quantized_decoder_with_past = nncf.quantize(
            ov_model.decoder_with_past.model,
            nncf.Dataset(decoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(decoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.95)
        )
        ov.save_model(quantized_decoder_with_past, quantized_distil_model_path / "openvino_decoder_with_past_model.xml")
        del quantized_decoder_with_past
        del decoder_calibration_data
        gc.collect()

        # Copy the config file and the first-step-decoder manually
        shutil.copy(distil_model_path / "config.json", quantized_distil_model_path / "config.json")
        shutil.copy(distil_model_path / "openvino_decoder_model.xml", quantized_distil_model_path / "openvino_decoder_model.xml")
        shutil.copy(distil_model_path / "openvino_decoder_model.bin", quantized_distil_model_path / "openvino_decoder_model.bin")

    quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(quantized_distil_model_path, compile=False)
    quantized_ov_model.generation_config = ov_model.generation_config
    quantized_ov_model.compile()
    return quantized_ov_model



if not distil_model_path.exists():
    ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
        distil_model_id, export=True, compile=False
    )
    ov_distil_model.half()
    ov_distil_model.save_pretrained(distil_model_path)
else:
    ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
        distil_model_path, compile=False
    )

pt_distil_model = AutoModelForSpeechSeq2Seq.from_pretrained(distil_model_id)
pt_distil_model.eval()
ov_distil_model.generation_config = pt_distil_model.generation_config
del pt_distil_model
gc.collect()
ov_distil_model.compile()
#quantize
ov_distil_model=quantize(ov_distil_model)
gc.collect()

pipe = pipeline(
    "automatic-speech-recognition",
    model=ov_distil_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
)




def format_timestamp(seconds: float):
    """
    format time in srt-file expected format
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (
        f"{hours}:" if hours > 0 else "00:"
    ) + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def prepare_srt(transcription):
    """
    Format transcription into srt file format
    """
    segment_lines = []
    for idx, segment in enumerate(transcription["chunks"]):
        timestamps = segment["timestamp"]
        time_start = format_timestamp(timestamps[0])
        if(timestamps[1] is not None):
            time_end = format_timestamp(timestamps[1])
        else:
            time_end="None"
        
        time_str = f"{time_start} --> {time_end}\n"
        
        segment_lines.append(segment["text"])
    return segment_lines,time_str

