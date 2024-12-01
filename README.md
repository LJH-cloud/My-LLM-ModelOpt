```
pip install -e .
```
```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir ~/.cache/llama/llama-2-7b-chat/ \
    --tokenizer_path ~/.cache/llama/llama-2-7b-chat/tokenizer.model \
    --max_gen_len 512 --max_batch_size 8
```