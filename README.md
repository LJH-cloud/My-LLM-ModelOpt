# My LLM ModelOpt
## Set up
```
pip install -e .
```
## Run
```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir ~/.cache/llama/llama-2-7b-chat/ \
    --tokenizer_path ~/.cache/llama/llama-2-7b-chat/tokenizer.model \
    --max_gen_len 512 --max_batch_size 1
```
## Eval on MMLU, GSM8K, HumanEval
```
cd test

# mmlu
cd mmlu
tar -xzvf data.tar.gz
cd ..
torchrun --nproc_per_node 1 eval_mmlu.py \
    --ckpt_dir ~/.cache/llama/llama-2-7b-chat/ \
    --tokenizer_path ~/.cache/llama/llama-2-7b-chat/tokenizer.model \
    --ntrain 5

# gsm8k
torchrun --nproc_per_node 1 eval_gsm8k.py \
    --ckpt_dir ~/.cache/llama/llama-2-7b-chat/ \
    --tokenizer_path ~/.cache/llama/llama-2-7b-chat/tokenizer.model

# humaneval
torchrun --nproc_per_node 1 eval_humaneval.py \
    --ckpt_dir ~/.cache/llama/llama-2-7b-chat/ \
    --tokenizer_path ~/.cache/llama/llama-2-7b-chat/tokenizer.model
evaluate_functional_correctness human-eval/data/HumanEval_res.jsonl
```