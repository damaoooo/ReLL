from dataclasses import dataclass
from typing import Any, Dict, List
import unsloth
import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from transformers import Trainer, TrainingArguments
from rich.console import Console

# --- 引入我们项目中的其他模块 ---
# --- 核心修正：使用相对导入 ---
# 这告诉Python在当前包(src)内寻找这些模块
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
        # margin 是一个超参数，我们将在TrainingArguments中定义它
        self.loss_fct = TripletMarginLoss(margin=self.args.triplet_margin)

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        官方推荐的Qwen Embedding模型池化方法。
        它依赖于左填充 (padding_side='left')。
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            # 如果是左填充，直接取最后一个token的hidden state
            return last_hidden_states[:, -1]
        else:
            # 如果是右填充，需要找到每个序列最后一个非padding token的位置
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss 方法来计算 Triplet Loss。
        """
        # 1. 分别获取 anchor, positive, negative 的模型输出
        anchor_outputs = model(
            input_ids=inputs["anchor_input_ids"],
            attention_mask=inputs["anchor_attention_mask"],
            output_hidden_states=True,
        )
        positive_outputs = model(
            input_ids=inputs["positive_input_ids"],
            attention_mask=inputs["positive_attention_mask"],
            output_hidden_states=True,
        )
        negative_outputs = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
            output_hidden_states=True,
        )

        # 2. 使用官方推荐的 last_token_pool 方法提取 embedding
        # --- 核心修正：从 .hidden_states[-1] 获取最后一层隐藏状态 ---
        anchor_emb = self.last_token_pool(anchor_outputs.hidden_states[-1], inputs["anchor_attention_mask"])
        positive_emb = self.last_token_pool(positive_outputs.hidden_states[-1], inputs["positive_attention_mask"])
        negative_emb = self.last_token_pool(negative_outputs.hidden_states[-1], inputs["negative_attention_mask"])


        # 3. 对 embedding 进行 L2 归一化，这是官方推荐的步骤
        eps = 1e-8
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1, eps=eps)
        positive_emb = F.normalize(positive_emb, p=2, dim=1, eps=eps)
        negative_emb = F.normalize(negative_emb, p=2, dim=1, eps=eps)

        # 4. 计算 Triplet Loss
        loss = self.loss_fct(anchor_emb, positive_emb, negative_emb)

        # 5. 返回 Hugging Face Trainer 期望的格式
        return (loss, {"loss": loss}) if return_outputs else loss

# 我们还需要扩展 TrainingArguments 来包含我们的自定义参数，以便在脚本中方便地设置
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
    console.rule("[bold yellow]开始调试 Trainer 流程[/bold yellow]")
    try:
        # --- 使用虚拟配置进行测试 ---
        # !!! 重要: 请将下面的路径替换为您自己脚本生成的真实路径 !!!
        dataset_pool_path = "data/processed_dataset/train_dataset_pool"
        positive_map_path = "data/processed_dataset/train_positive_map.pkl"
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        
        # --- 核心修正：保持序列长度，但严格限制批量大小 ---
        max_length = 2048 # 保持与计划的训练配置一致
        debug_batch_size = 1 # 关键：将批量大小降至1以在调试时避免OOM
        
        # --- 1. 加载模型和Tokenizer ---
        console.log("[bold]1. 加载模型和Tokenizer...[/bold]")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            lora_config_dict={
                "r": 8, "lora_alpha": 16, "lora_dropout": 0,
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
        # 将数据移动到模型所在的设备
        dummy_batch = {k: v.to(model.device) for k, v in dummy_batch.items()}
        console.log(f"虚拟批次创建成功 (大小={debug_batch_size}) 并已移动到GPU。")

        # --- 4. 测试 compute_loss ---
        console.log("\n[bold]4. 测试 compute_loss 方法...[/bold]")
        
        # --- 核心修正 ---
        # 实例化一个真实的 TripletTrainingArguments。
        # 关键：启用梯度检查点 (gradient_checkpointing) 来节省显存。
        training_args = TripletTrainingArguments(
            output_dir="./tmp_debug_trainer", # 这是一个必需的临时目录
            triplet_margin=1.0,
            gradient_checkpointing=True, # <--- 启用此项可以极大减少显存占用
            gradient_checkpointing_kwargs={"use_reentrant": True}, # 推荐用于新版PyTorch
            label_names=[] # <--- 明确告知Trainer我们没有标签列，以消除警告
        )
        
        trainer = TripletTrainer(model=model, args=training_args)
        
        # 手动调用compute_loss
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
