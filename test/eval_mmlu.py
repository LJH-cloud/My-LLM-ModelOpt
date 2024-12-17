import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoTokenizer,AutoModelForCausalLM
import time
from tqdm import tqdm
from llama import Llama
import logger
choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(
    args, 
    subject, 
    generator: Llama, 
    dev_df, 
    test_df
):
    model = generator.model
    tokenizer = generator.tokenizer

    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0]), desc=f'Processing items', leave=False):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        prompt_tokens = tokenizer.encode(prompt.strip(), bos=True, eos=False)

        while len(prompt_tokens) > 4000:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            prompt_tokens = tokenizer.encode(prompt.strip(), bos=True, eos=False)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # logits = model(
        #     input_ids=input_ids,
        # ).logits[:,-1].flatten()

        tokens = torch.full((1, len(prompt_tokens)), tokenizer.pad_id, dtype=torch.long, device="cuda")
        tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")
        logits, _ = model.forward(tokens, 0, None, 0)
        # logits = model.forward(tokens, 0)
        logits = logits[:,-1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer.encode("A", bos=False, eos=False)[-1]],
                        logits[tokenizer.encode("B", bos=False, eos=False)[-1]],
                        logits[tokenizer.encode("C", bos=False, eos=False)[-1]],
                        logits[tokenizer.encode("D", bos=False, eos=False)[-1]],
                        # logits[tokenizer("A").input_ids[-1]],
                        # logits[tokenizer("B").input_ids[-1]],
                        # logits[tokenizer("C").input_ids[-1]],
                        # logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    logger.log("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    logger.init_logger('log.txt')
    # tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=False,add_bos_token=False, model_max_length=4096,padding_side="right",trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(args.model,  torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
    )
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc="Processing subjects"):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, generator, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        logger.log("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        logger.log("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    logger.log("Average accuracy: {:.3f}".format(weighted_acc))
    logger.close_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    # parser.add_argument("--ngpu", "-g", type=int, default=8)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--ckpt_dir", "-m", type=str)
    parser.add_argument("--tokenizer_path", "-t", type=str)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--model", type=str, default='Llama')
    args = parser.parse_args()
    main(args)