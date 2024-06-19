#!/usr/bin/env python3




import transformers
import sys
import time
import os
from transformers.pipelines.pt_utils import KeyDataset
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import datasets
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

parser = argparse.ArgumentParser(description='Process some flags.')

# Add the arguments with the flags
parser.add_argument('-B', type=str, help='Flag B argument')
parser.add_argument('-I', type=str, help='Flag I argument')
parser.add_argument('-O', type=str, help='Flag O argument')

# Parse the arguments
args = parser.parse_args()

# Retrieve and print the values
BATCH_SIZE = args.B
INPUT_LEN = args.I
OUT_LEN = args.O

if BATCH_SIZE == None:
    BATCH_SIZE = 1
else:
    BATCH_SIZE = int(BATCH_SIZE)

if INPUT_LEN == None:
    INPUT_LEN = 1
else:
    INPUT_LEN = int(INPUT_LEN)

if OUT_LEN == None:
    OUT_LEN = 2
else:
    OUT_LEN = int(OUT_LEN)    


print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'INPUT_LEN:  {INPUT_LEN}')
print(f'OUTPUT_LEN: {OUT_LEN}')
print("="*20)



# To use meta-llama model
os.environ['HF_TOKEN'] = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Load Dataset and Limit to 512 inputs
dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
requests = [req for req in KeyDataset(dataset, "text")[:512]]

# Set number of input tokens per request
temp = []
for data in requests:
    words = data.split(" ")[:INPUT_LEN]
    word = " ".join(words)
    temp.append(word)
requests = temp


# ???
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)



#############################################################################################################################################

# Running LLAMA. Do not touch!
 
model_id = "meta-llama/Meta-Llama-3-8B"

# Always have device_map="auto" to solve issue of CUDA malloc no memory.
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

start_time = time.time()


# # Unique Profiling inside or none at all
res = pipeline(requests[:BATCH_SIZE], batch_size=BATCH_SIZE, max_new_tokens=OUT_LEN)

end_time = time.time()

# #############################################################################################################################################


for r in res:
    print(r)


print("--- %s seconds ---" % (end_time - start_time))

