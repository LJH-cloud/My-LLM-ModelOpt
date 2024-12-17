import argparse
import tqdm
import torch
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from llama import Llama2

"""
git clone https://github.com/openai/human-eval
$ pip install -e human-eval
evaluate_functional_correctness sample-output-file
"""


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("def ")[0]
        sents.append(sent)
    return sents


def generate_sample(generator: Llama2, input_txt):
    # input_ids = tokenizer.tokenizer.encode(input_txt)
    # raw_text_len = len(input_ids)
    # context_enc = torch.tensor([input_ids]).to(model.device)
    # print(f"Input text: {input_txt}\n")
    # outputs = model.generate(context_enc)
    # output_text = decode(outputs, tokenizer, raw_text_len)[0]
    # print(f"\nOutput text: \n{output_text}\n")
    # output_text, _ = generator.text_completion(prompts=[input_txt])
    output_text = generator.text_completion(prompts=[input_txt])
    output_text = output_text[0]['generation'].split("\n\n\n")[0].split("\n\n")[0].split("def")[0]
    print(f"Input text: {input_txt}")
    print(f"Output text: {output_text}")
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("--data_dir", "-d", type=str, default="human-eval/data/HumanEval.jsonl")
    parser.add_argument("--save_dir", "-s", type=str, default="human-eval/data/HumanEval_res_origin.jsonl")
    parser.add_argument("--ckpt_dir", "-m", type=str)
    parser.add_argument("--tokenizer_path", "-t", type=str)
    parser.add_argument("--max_seq_len", type=int, default=4096)

    args = parser.parse_args()

    generator = Llama2.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
    )

    f_output = jsonlines.Writer(open(args.save_dir, "w", encoding="utf-8"))

    f = jsonlines.open(args.data_dir)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc="task_idx"):
            prompt = jobj["prompt"]
            task_id = jobj["task_id"]
            gen_sents = generate_sample(generator, prompt)
            gen_jobjs = {"task_id": task_id, "completion": gen_sents}
            output.write(gen_jobjs)
    f_output.close()