# Test
测试时使用单GPU，显存需要48GB+
## MMLU
输入mmlu文件夹内的数据集，输出为各类别的精准度，在mmlu文件夹下的results以及log.txt中
## GSM8K
输入gsm8k文件夹下的gsm8k_prompt.txt提示词以及gsm8k_test.jsonl数据集，输出为各数学问题以及模型对应回答gsm8k_results.jsonl
## HumanEval
输入humaneval/data文件夹下的数据集，输出humaneval/data文件夹下的代码生成results结果