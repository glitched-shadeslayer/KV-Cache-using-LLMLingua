import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BartTokenizer

def compare_input_tokens(og_text, summarized_text, tokenizer):
    og_tokens = tokenizer(og_text, return_tensors="pt")
    og_token_size = og_tokens['input_ids'].shape[1]
    summarized_tokens = tokenizer(summarized_text, return_tensors="pt")
    summarized_token_size = summarized_tokens['input_ids'].shape[1]
    return summarized_token_size / og_token_size * 100

if __name__ == '__main__':
    ratios = [0.5, 0.6, 0.7]
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    compression_arr = {}

    for ratio in ratios:
        file_path = f"summarize_{int(ratio * 100)}.csv"
        print(file_path)
        print("Ratio = ", ratio)
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            val = compare_input_tokens(row['context'], row['summary'], tokenizer)
            if ratio not in compression_arr:
                compression_arr[ratio] = []
            compression_arr[ratio].append(val)

        plt.plot(compression_arr[ratio], label=ratio)
        plt.legend()
        plt.show()
        print(sum(compression_arr[ratio]) / len(compression_arr[ratio]))

