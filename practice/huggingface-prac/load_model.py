from transformers import pipeline
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

pipe=pipeline("text-generation",model="Qwen/Qwen3-4B-Instruct-2507")
messages=[
    {"role":"user","content":"你好，你是谁？"}
]
pipe(messages)