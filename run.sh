#!/bin/bash

# 默认值
MODEL=""
INPUT_DIR=""
ENABLE_OPTIMIZE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL="$2"
            shift # 跳过参数值
            shift # 跳过参数名称
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift
            shift
            ;;
        --enable-optimize)
            ENABLE_OPTIMIZE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查参数是否完整
if [[ -z "$MODEL" || -z "$INPUT_DIR" ]]; then
    echo "使用方法: $0 --model <模型类型> --input-dir <模型位置> [--enable-optimize]"
    exit 1
fi

# 输出参数
echo "模型类型: $MODEL"
echo "模型位置: $INPUT_DIR"
if $ENABLE_OPTIMIZE; then
    echo "量化: 已启用"
else
    echo "量化: 未启用"
fi

# 执行相应操作
if [[ "$MODEL" == "llama" && $ENABLE_OPTIMIZE == false ]]; then
    echo "执行 LLaMA 非量化模式..."
    COMMAND="torchrun --nproc_per_node 1 llama/example_chat_completion.py \\
        --ckpt_dir $INPUT_DIR \\
        --tokenizer_path $INPUT_DIR/tokenizer.model \\
        --max_gen_len 512 --max_batch_size 1"
    echo "运行命令: $COMMAND"
    eval $COMMAND
elif [[ "$MODEL" == "llama" && $ENABLE_OPTIMIZE == true ]]; then
    echo "执行 LLaMA 量化模式..."
    COMMAND="python llama/generate.py --model_name_or_path $INPUT_DIR --enable_streaming"
    echo "运行命令: $COMMAND"
    eval $COMMAND
elif [[ "$MODEL" == "qwen" ]]; then
    echo "执行 Qwen 模式..."
    COMMAND="python qwen/generate.py --model_name_or_path $INPUT_DIR --enable_streaming"
    echo "运行命令: $COMMAND"
    eval $COMMAND
else
    echo "模型未知"
fi
