#!pip install transformers==4.41.2
# Important to install above transformer version, unless error occurs
#See link: https://github.com/THUDM/CogVLM2/issues/181#issuecomment-2381807778

# !pip install sentencepiece datasets evaluate
# !pip install llmlingua

import torch
import json
import time
import os
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from tqdm.notebook import tqdm 
import numpy as np
#from sklearn.metrics import f1_score
#from LLMLingua.experiments.llmlingua2.evaluation.metrics import qa_f1_score
from metrics import qa_f1_score
if 'model' in globals():
    del model 
torch.cuda.empty_cache()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#######################################
print('cuda is',torch.cuda.is_available()) 
#################################################
# 3 model & dataset 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ChatGLM-6B
model = AutoModel.from_pretrained(
    "THUDM/chatglm2-6b-32k",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=True
).to(device)

model=model.eval()
#print('model config is:',model.generation_config)

####################################################
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)#.to(device)


####################################################
# Monkey-patch the tokenizer's _pad method to handle the padding_side argument
original_pad = tokenizer._pad
def new_pad(self, encoded_inputs, max_length=None, padding_strategy="longest", pad_to_multiple_of=None, return_attention_mask=None, **kwargs):
    return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)
tokenizer._pad = new_pad.__get__(tokenizer, type(tokenizer))



##########################################################
# Load the "multifieldqa_en" subset
dataset_name = "multifieldqa_en"
data = load_dataset("THUDM/LongBench", dataset_name, split="test", cache_dir="custom_cache_dir")

# print(type(data))#should be <class 'datasets.arrow_dataset.Dataset'>
# print(data[0])  #keys like "context", "question", "answers"

########################################################
#RANDOM token PRUNE
def random_prune(text, compression_ratio=0.5):

    tokens = tokenizer.tokenize(text).to(device)  # e.g., ["The", " quick", " brown"...]
    
    num_keep = int(len(tokens) * (compression_ratio)) # compression ratio means remaining 
    
    # Step 3: Randomly pick tokens (like lottery balls)
    kept_indices = sorted(  # Maintain original order
        np.random.choice(  # Random selection
            len(tokens), 
            num_keep, 
            replace=False  # No duplicates
        )
    )
    
    # Step 4: Rebuild text from kept tokens
    pruned_tokens = [tokens[i] for i in kept_indices]
    return tokenizer.convert_tokens_to_string(pruned_tokens)  # Tokens ➔ text


##############################################################################
# LINGUA COMPRESSION

from llmlingua import PromptCompressor

def compress_with_llmlingua(context, question, ratio=0.5, dynamic_context_compression_ratio=0.3):
    #print(type(PromptCompressor()))
    compressor = PromptCompressor()#.to(device)
    compressed_context = compressor.compress_prompt(
        context,  
        question = question, 
        rate=ratio, 
        condition_in_question="after_condition",
        reorder_context="sort",
        dynamic_context_compression_ratio = dynamic_context_compression_ratio,
        condition_compare=True,
        context_budget="+100",
        rank_method="longllmlingua",
        )#, device=device)
    # compressed_context = compressor.compress_prompt_llmlingua2(context, rate=ratio)
    '''
    token_count = len(context.split())
    #compressed_token_count = len(compressed_context.split())
    file_name = "compressed_context_info.txt"

# 检查文件是否已存在
    if os.path.exists(file_name):
        print("文件已存在，程序终止。")
        exit(1)

# 写入文件
    with open(file_name, "w", encoding="utf-8") as f:
        #f.write(f"Type of context: {type(context)}\n")
        #f.write(f"context: {context}\n")
        f.write(f"ompressed_context: {compressed_context}\n")
        f.write(f"Token count: {token_count}\n")
        #f.write(f"Token count: {compressed_token_count}\n")

    print("信息已写入文件。")'''
    return compressed_context

# #  Using gpu
# def compress_with_llmlingua(context, ratio=0.5):
#     # Initialize with GPU and a small LM (e.g., GPT-2)
#     compressor = PromptCompressor(
#         model_name="gpt2",           # Use a smaller model for faster compression
#         #device="cuda",               # Use GPU, but lingua will automatically use gpu if available so commented out
#         use_llmlingua2=True          # Enable optimized GPU compression
#     )
    
#     # Compress with target ratio (e.g., 0.5 = keep 50% tokens)
#     compressed_context = compressor.compress_prompt(
#         context, 
#         rate=ratio#,                  # Target compression ratio
#         #force_tokens=True            # Ensure exact token count
#     )
    
#     return compressed_context["compressed_prompt"]

##############################################################################
# Any summarizer
from transformers import BartForConditionalGeneration, BartTokenizer
def compress_with_bart(context, ratio=0.5):
    model= BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer(
        context, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(device)
    # Calculate target length based on ratio
    input_length = inputs['input_ids'].shape[1]
    target_length = int(input_length * ratio)
    
    # Generate summary
    summary_ids = model.generate(
        inputs.input_ids,
        num_beams=4,
        max_length=target_length,
        min_length=max(target_length - 100, 50),  # Keep min_length within bounds
        length_penalty=2.0,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# pip install bert-extractive-summarizer==0.4.2 <- compatitable with transformer 4.41.2

# from gensim.summarization import summarize
# from summarizer import Summarizer

# def compress_with_bert(context, ratio=0.5):
#     """
#     Compresses text using BERT embeddings.
#     ratio=0.5 → keeps 50% of sentences.
#     """
#     model = Summarizer()
#     summary = model(
#         context,
#         ratio=ratio,           # Target compression ratio (0.1-0.9)
#         min_length=10,         # Minimum sentence length to keep
#         use_gpu=False          # Set to True if GPU available
#     )
#     return summary

# def compress_with_textrank(context, ratio=0.5):
#     """
#     Adjustable compression with TextRank.
#     ratio=0.5 → summary is 50% of original length.
#     """
#     try:
#         return summarize(context, ratio=ratio)
#     except ValueError:  # Fallback for very short texts
#         return context


################################################################################
#Baseline

def no_pruning(context: str) -> str:
    """Returns the full context (no pruning)."""
    return context


################################################################################
# from sklearn.metrics import f1_score
from collections import Counter

def compute_f1(reference, prediction, **kwargs):
    """ Compute F1 score for answer evaluation """
    common=Counter(prediction) & Counter(reference)
    num_same=sum(common.values())
    if num_same==0:
        return 0
    precision=1.0*num_same/len(prediction)
    recall=1.0*num_same/len(reference)

    return 2 *(precision * recall) / (precision + recall)

# def compute_f1(reference, prediction):
#     """ Compute F1 score for answer evaluation """
#     ref_tokens = set(reference.split())
#     pred_tokens = set(prediction.split())

#     common_tokens =ref_tokens&pred_tokens
#     precision=len(common_tokens)/len(pred_tokens) if pred_tokens else 0
#     recall=len(common_tokens)/len(ref_tokens) if ref_tokens else 0

#     if precision+ recall == 0:
#         return 0.0

#     return 2 *(precision * recall) / (precision + recall)

# def measure_latency(func, *args):
#     start = time.time()
#     func(*args)
#     return time.time() - start

# def get_memory_usage():
#     return torch.cuda.memory_allocated() / (1024 ** 2)  # in MB


##################################################################################
def evaluate(data, compression_func, ratio=None):
    f1_scores = []
    latencies = []
    memory_usages = []
    peak_memory_usages = []
    
    #for idx in range(10):  # Test on 10 examples
    for idx in range(len(data)):  # Test on all examples
        example = data[idx] 
        #print(type(example))
        #print(example)
        context = example["context"]
        question = example["input"]
        answers =example["answers"]
        if ratio is not None:
            compressed_context = compression_func(context, question, ratio)
        else:
            compressed_context = compression_func(context)  #for no.prune & compressr
        
        # test_response, _ = model.chat(
        # tokenizer,
        # "What is the capital of France?",
        # history=[("European geography", "Answer:")]
        # )
        # print(test_response)  # Should return "Paris"
        
        prompt = f"Context: {compressed_context}\nQuestion: {question}"
        #prompt =prompt.lstrip("\n").split("\n")[0]
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        #response, _ = model.chat(tokenizer, prompt)
        response, _ = model.chat(tokenizer, compressed_context)
        response = response.lstrip("\n").split("\n")[0]
        latency = time.time() - start_time
        
        # Measure memory
        memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        #Measure peak memory?
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Calculate F1 score
        #f1 = f1_score([answers], [response], average="macro")  # Stored f1score function is not appropritate
        #f1=compute_f1(answers, response)
        f1 = 0.0
        for ground_truth in answers:
            f1 = max(
                f1,
                qa_f1_score(prediction=response, ground_truth=ground_truth)
            )
        #f1 = qa_f1_score(prediction=response, ground_truth=answers)
        f1_scores.append(f1)
        latencies.append(latency)
        memory_usages.append(memory)
        peak_memory_usages.append(peak_memory)

        # print(f"Response: {response}")
        # print(f"F1 Score: {f1:.2f}, Latency: {latency:.2f}s, Memory: {memory:.2f}MB\n")
    
    return {
        "avg_f1": np.mean(f1_scores),
        "avg_latency": np.mean(latencies),
        #"avg_memory": np.mean(memory_usages),
        "avg_peak_memory": np.mean(peak_memory_usages)
    }
#####################################################################################
with open("results.txt", "w") as f:
    f.write("=== Compression Benchmark Results ===\n\n")

# Reusable function to print and save results
def log_result(message):
    #print(message)
    with open("results.txt", "a") as f:  # 'a' = append mode
        f.write(message + "\n")
        
        
####################################################    
pruning_ratios = [0.7]#, 0.6, 0.7]


for ratio in pruning_ratios:
    lingua_results = evaluate(data, compress_with_llmlingua, ratio=ratio)
    
    print(f"\n=== llmlingua compression ({int(ratio*100)}%) ===")
    print(f"F1: {lingua_results['avg_f1']:.2f}, Latency: {lingua_results['avg_latency']:.2f}s,  Peak Memory: {lingua_results['avg_peak_memory']:.2f} MB")
    log_result(f"\n=== llmlingua compression ({int(ratio*100)}%) ===")
    log_result(f"F1: {lingua_results['avg_f1']:.2f}, Latency: {lingua_results['avg_latency']:.2f}s,  Peak Memory: {lingua_results['avg_peak_memory']:.2f} MB")
