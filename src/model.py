import logging
from typing import Dict, Tuple

import torch
from rich.console import Console
# --- 核心改动：引入 Unsloth ---
from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizer, PreTrainedModel

# 使用Rich打印美观的日志信息
console = Console()

def load_model_and_tokenizer(
    model_name: str,
    lora_config_dict: Dict,
    use_gradient_checkpointing: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    使用 Unsloth 加载、量化并应用LoRA到指定的预训练模型。

    Args:
        model_name (str): 要加载的Hugging Face模型名称。
        lora_config_dict (Dict): 包含LoRA参数的字典 (r, lora_alpha等)。
        use_gradient_checkpointing (bool): 是否启用Unsloth优化的梯度检查点。

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: 配置好的模型和分词器。
    """
    console.rule(f"[bold blue]使用 Unsloth 加载模型: {model_name}[/bold blue]")

    # --- 1. 使用 Unsloth 加载基础模型 ---
    # Unsloth 会自动处理4-bit量化、数据类型选择和Flash Attention等所有优化
    logging.info("正在使用 Unsloth 加载 4-bit 量化模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=None,  # None 会让Unsloth自动选择最佳数据类型
        load_in_4bit=True,
        trust_remote_code=True,
    )
    logging.info("基础模型加载完成。")

    # --- 2. 使用 Unsloth 应用 LoRA 配置 ---
    logging.info("正在应用 Unsloth LoRA 适配器...")
    
    # 关键：使用Unsloth优化的梯度检查点可以极大节省显存
    # Unsloth推荐的梯度检查点参数值为 "unsloth"
    grad_checkpointing_mode = "unsloth" if use_gradient_checkpointing else False

    model = FastLanguageModel.get_peft_model(
        model,
        **lora_config_dict, # 将配置文件中的所有LoRA参数解包传入
        use_gradient_checkpointing=grad_checkpointing_mode,
        random_state=3407,
    )
    
    logging.info("Unsloth LoRA应用完成。")
    console.log("可训练参数详情:")
    model.print_trainable_parameters()

    # --- 3. 处理 Tokenizer 的 Padding 设置 ---
    # 这一步保持不变，因为我们的 `last_token_pool` 逻辑依赖于左填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        logging.info("Tokenizer的pad_token已设置为eos_token。")
    
    # 确保padding_side是left
    tokenizer.padding_side = "left"
    logging.info("Tokenizer的padding_side已设置为'left'。")

    return model, tokenizer


def debug_unsloth_model_loading():
    """
    一个用于调试的函数，测试Unsloth模型加载、LoRA应用和前向传播是否正常。
    """
    console.rule("[bold yellow]开始调试 Unsloth 模型加载流程[/bold yellow]")
    try:
        # --- 使用虚拟配置进行测试 ---
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        lora_config_dict = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
        }

        # --- 加载模型 ---
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            lora_config_dict=lora_config_dict,
            use_gradient_checkpointing=True
        )

        # --- 创建虚拟输入 ---
        logging.info("创建虚拟输入并进行前向传播测试...")
        dummy_instruction = "Represent this LLVM IR for searching for similar functions:"
        dummy_llvm_ir = """
        define i32 @main() {
          ret i32 0
        }
        """
        dummy_text = dummy_instruction + dummy_llvm_ir
        
        inputs = tokenizer(dummy_text, return_tensors="pt")
        
        # 将输入移动到模型所在的设备 (GPU)
        # Unsloth模型加载时已在GPU上，所以这一步是确认
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        logging.info(f"虚拟输入已移动到设备: cuda")

        # --- 前向传播 ---
        with torch.no_grad(): # 在推理/调试时不需要计算梯度
            outputs = model(**inputs)
        
        logging.info("前向传播成功！")
        console.log(f"模型输出 (last_hidden_state) 的形状: [bold green]{outputs.last_hidden_state.shape}[/bold green]")
        console.rule("[bold green]调试成功！Unsloth模型可以在您的GPU上正常运行。[/bold green]")

    except Exception as e:
        console.print_exception()
        console.rule("[bold red]调试失败！请检查上面的错误信息。[/bold red]")


if __name__ == "__main__":
    # 当直接运行此脚本时，执行调试函数
    debug_unsloth_model_loading()
