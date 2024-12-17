import re
import torch
import argparse
import json
import jsonlines
import numpy as np
from tqdm import tqdm

from llama import Llama


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(generator: Llama, input_txt):
    # input_ids = tokenizer.tokenizer.encode(input_txt)
    # raw_text_len = len(input_ids)
    # context_enc = torch.tensor([input_ids]).to(model.device)
    # print(f"Input text: {input_txt}\n")
    # outputs = model.generate(context_enc)
    # output_text = decode(outputs, tokenizer, raw_text_len)[0]
    # print(f"\nOutput text: {output_text}\n")
    output_text, _ = generator.text_completion(prompts=[input_txt])
    # output_text = generator.text_completion(prompts=[input_txt])
    output_text = output_text[0]['generation'].split("\n\n\n")[0].split("\n\n")[0].split("Question:")[0]
    print(f"Input text: {input_txt}")
    print(f"Output text: {output_text}")
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\-?[0-9\.\,]+", completion)[-1].strip().replace(",", "")
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    pred = extract_answer(completion)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    print(pred, gold)
    return extract_answer(completion) == gold, pred, gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="gsm8k_test.jsonl")
    parser.add_argument("--save_dir", "-s", type=str, default="gsm8k_results.jsonl")
    parser.add_argument("--ckpt_dir", "-m", type=str)
    parser.add_argument("--tokenizer_path", "-t", type=str)
    parser.add_argument("--max_seq_len", type=int, default=4096)

    args = parser.parse_args()

    fewshot_prompt = open("gsm8k_prompt.txt").read()
    # if args.sample_input_file is not None:
    #     dataset = load_from_disk(args.sample_input_file)
    # else:
    #     config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    #     dataset = load_dataset("gsm8k", "main", download_config=config)

    # test = dataset["test"]
    test = []
    with open(args.data_dir, 'r') as f:
        for line in f:
            test.append(json.loads(line))

    # print("Loading tokenizer ...")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.checkpoint_path, trust_remote_conde=True
    # )

    # print("Loading model ...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.checkpoint_path, device_map="auto", trust_remote_code=True
    # ).eval()
    # model.generation_config = GenerationConfig.from_pretrained(
    #     args.checkpoint_path, trust_remote_code=True
    # )
    # model.generation_config.do_sample = False

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
    )

    f_output = jsonlines.Writer(open(args.save_dir, "w", encoding="utf-8"))
    acc_res = []
    for doc in tqdm(test, desc='Processing sentence'):
        context = doc_to_text(doc)
        completion = generate_sample(generator, context)
        answer = doc["answer"]
        acc, pred, gold = is_correct(completion, answer)
        doc["completion"] = completion
        doc["pred"] = pred
        doc["gold"] = gold
        doc["acc"] = acc
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print("Acc: ", np.mean(acc_res))