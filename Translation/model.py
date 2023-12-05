from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import openvino as ov

import torch
import torch.onnx

model_id="Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

translator = pipeline("translation", 
                      model=model,
                      tokenizer=tokenizer,
                      )






