import logging
from typing import Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

# 使用Rich打印美观的日志信息
console = Console()

def load_model_and_tokenizer(
    model_name: str,
    quantization_bits: int,
    lora_config_dict: Dict,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    加载、量化并应用PEFT(LoRA)到指定的预训练模型。

    Args:
        model_name (str): 要加载的Hugging Face模型名称。
        quantization_bits (int): 模型的量化位数 (例如 4 或 8)。
        lora_config_dict (Dict): 包含LoRA参数的字典。

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: 配置好的模型和分词器。
    """
    console.rule(f"[bold blue]加载模型和分词器: {model_name}[/bold blue]")

    # --- 1. 配置量化 ---
    bnb_config = None
    if quantization_bits == 4:
        logging.info("正在配置 4-bit 量化 (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization_bits == 8:
        logging.info("正在配置 8-bit 量化 (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # --- 2. 加载模型 ---
    logging.info("正在加载基础模型...")
    # 使用 AutoModel 加载基础模型，不带任务头
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" # 仍然可以尝试开启Flash Attention
    )
    logging.info("基础模型加载完成。")

    # --- 3. 加载Tokenizer ---
    logging.info("正在加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        logging.info("Tokenizer的pad_token已设置为eos_token。")

    # --- 4. 应用PEFT (LoRA) ---
    if lora_config_dict:
        logging.info("正在应用PEFT (LoRA) 配置...")
        if quantization_bits in [4, 8]:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(**lora_config_dict)
        model = get_peft_model(model, lora_config)
        
        logging.info("LoRA应用完成。")
        console.log("可训练参数详情:")
        model.print_trainable_parameters()
    else:
        logging.warning("未提供LoRA配置，将进行全量微调（不推荐）。")

    return model, tokenizer


def debug_model_loading():
    """
    一个用于调试的函数，测试模型加载、量化、LoRA应用和前向传播是否正常。
    """
    console.rule("[bold yellow]开始调试模型加载流程[/bold yellow]")
    try:
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        quantization_bits = 4
        lora_config_dict = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        }

        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            quantization_bits=quantization_bits,
            lora_config_dict=lora_config_dict
        )

        dummy_text = "Represent this LLVM IR for searching for similar functions: define i32 @main() { ret i32 0 }"
        inputs = tokenizer(dummy_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logging.info("前向传播成功！")
        console.log(f"模型输出 (last_hidden_state) 的形状: [bold green]{outputs.last_hidden_state.shape}[/bold green]")
        console.rule("[bold green]调试成功！模型可以在您的GPU上正常运行。[/bold green]")

    except Exception as e:
        console.print_exception()
        console.rule("[bold red]调试失败！请检查上面的错误信息。[/bold red]")


if __name__ == "__main__":
    debug_model_loading()
