from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from transformers import Trainer, TrainingArguments
from rich.console import Console

# --- 引入我们项目中的其他模块 ---
from .dataset import OnlineTripletDataset, TripletDataCollator
from .model import load_model_and_tokenizer


# 使用Rich打印美观的日志信息
console = Console()


class TripletTrainer(Trainer):
    """
    一个自定义的Trainer，用于计算Triplet Loss。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = TripletMarginLoss(margin=self.args.triplet_margin)

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        官方推荐的Qwen Embedding模型池化方法。
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss 方法来计算 Triplet Loss。
        """
        # --- 核心回滚：AutoModel的输出不包含 .hidden_states，而是直接提供 .last_hidden_state ---
        anchor_outputs = model(
            input_ids=inputs["anchor_input_ids"],
            attention_mask=inputs["anchor_attention_mask"],
        )
        positive_outputs = model(
            input_ids=inputs["positive_input_ids"],
            attention_mask=inputs["positive_attention_mask"],
        )
        negative_outputs = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
        )

        anchor_emb = self.last_token_pool(anchor_outputs.last_hidden_state, inputs["anchor_attention_mask"])
        positive_emb = self.last_token_pool(positive_outputs.last_hidden_state, inputs["positive_attention_mask"])
        negative_emb = self.last_token_pool(negative_outputs.last_hidden_state, inputs["negative_attention_mask"])

        eps = 1e-8
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1, eps=eps)
        positive_emb = F.normalize(positive_emb, p=2, dim=1, eps=eps)
        negative_emb = F.normalize(negative_emb, p=2, dim=1, eps=eps)

        loss = self.loss_fct(anchor_emb, positive_emb, negative_emb)

        return (loss, {"loss": loss}) if return_outputs else loss

@dataclass
class TripletTrainingArguments(TrainingArguments):
    """
    为Triplet Loss训练添加自定义参数。
    """
    triplet_margin: float = 1.0


def debug_trainer():
    """
    一个用于调试的函数，测试整个训练流程（模型、数据、训练器）是否能协同工作。
    """
    console.rule("[bold yellow]开始调试 Trainer 流程 (Hugging Face PEFT版)[/bold yellow]")
    try:
        # --- 使用虚拟配置进行测试 ---
        dataset_pool_path = "data/processed_dataset/train_dataset_pool"
        positive_map_path = "data/processed_dataset/train_positive_map.pkl"
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        max_length = 2048
        debug_batch_size = 1
        
        # --- 1. 加载模型和Tokenizer ---
        console.log("[bold]1. 加载模型和Tokenizer...[/bold]")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            quantization_bits=4,
            lora_config_dict={
                "r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "bias": "none"
            }
        )

        # --- 2. 加载数据集和DataCollator ---
        console.log("\n[bold]2. 加载数据集和DataCollator...[/bold]")
        train_dataset = OnlineTripletDataset(
            dataset_pool_path=dataset_pool_path,
            positive_map_path=positive_map_path
        )
        data_collator = TripletDataCollator(
            tokenizer=tokenizer,
            max_length=max_length
        )

        # --- 3. 准备一个批次的数据 ---
        console.log("\n[bold]3. 准备一个虚拟批次...[/bold]")
        dummy_batch = data_collator([train_dataset[i] for i in range(debug_batch_size)])
        dummy_batch = {k: v.to(model.device) for k, v in dummy_batch.items()}
        console.log(f"虚拟批次创建成功 (大小={debug_batch_size}) 并已移动到GPU。")

        # --- 4. 测试 compute_loss ---
        console.log("\n[bold]4. 测试 compute_loss 方法...[/bold]")
        training_args = TripletTrainingArguments(
            output_dir="./tmp_debug_trainer",
            triplet_margin=1.0,
            gradient_checkpointing=True,
            label_names=[],
            remove_unused_columns=False
        )
        
        trainer = TripletTrainer(model=model, args=training_args)
        
        loss = trainer.compute_loss(model, dummy_batch)
        console.log(f"成功计算损失！ Loss: [bold green]{loss.item()}[/bold green]")
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "Loss必须是一个标量张量"

        console.rule("[bold green]调试成功！Trainer 核心逻辑工作正常。[/bold green]")

    except FileNotFoundError:
        console.print(f"[bold red]错误: 找不到数据文件。[/bold red]")
        console.print(f"请确保以下路径是正确的，并且您已经运行了 '03_split_dataset.py' 脚本:")
        console.print(f"  - 数据集池: '{dataset_pool_path}'")
        console.print(f"  - 正样本映射: '{positive_map_path}'")
    except Exception as e:
        console.print_exception()
        console.rule("[bold red]调试失败！请检查上面的错误信息。[/bold red]")


if __name__ == "__main__":
    # 当直接运行此脚本时，执行调试函数
    debug_trainer()
