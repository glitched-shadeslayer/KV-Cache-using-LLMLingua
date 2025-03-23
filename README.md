# KV Cache Compression using LLMLingua

This repository has collection of experiments over three approaches for KV Cache compression: randomized token pruning, summarization, and LLMLingua.
Check out the following files:
- summarization_model_experiment has files related to running the bart-large-cnn over all the datasets. `summarize_longbench_using_bart_cnn.ipynb` takes dataset_name and generates n-number of compressed contexts and stores them in a csv files stored in the respective dataset's folder. 
- `summarization_model_experiment/chatglm_inference_for_summarized.ipynb` performs inference using chatglm model for bart-summarized prompts and generates .
- All the metrics are generated in `comparing_metrics.ipynb`
- Experiments for the randomized token pruning and no-context processing are performed in `llimlinga_randprune_noprune.py` and `SJHa_226_Random_and_no_prune.ipynb`.
- LLMLingua experiments can be found in `multiple_answers_longllmlingua.py` and `SJHa_226_llimlinga.ipynb`