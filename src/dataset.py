import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_from_disk
from rich.console import Console
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# 使用Rich打印美观的日志信息
console = Console()


class OnlineTripletDataset(TorchDataset):
    """
    一个用于对比学习的PyTorch数据集。
    它在每次被调用时，动态地从数据池中采样一个 (Anchor, Positive, Negative) 三元组。
    """
    def __init__(self, dataset_pool_path: str, positive_map_path: str, steps_per_epoch: Optional[int] = None):
        """
        初始化数据集。

        Args:
            dataset_pool_path (str): 数据集池路径。
            positive_map_path (str): 正样本映射.pkl文件路径。
            steps_per_epoch (Optional[int]): 【关键】定义一轮的步数。如果为None，则一轮会遍历所有锚点。
        """
        super().__init__()
        self.dataset_pool = load_from_disk(dataset_pool_path)
        console.log(f"加载自 [cyan]{dataset_pool_path}[/cyan] 的数据集池，共 {len(self.dataset_pool):,} 个函数。")

        with open(positive_map_path, 'rb') as f:
            self.positive_map = pickle.load(f)
        console.log(f"加载自 [cyan]{positive_map_path}[/cyan] 的正样本映射。")
        
        self.anchor_indices = sorted(list(self.positive_map.keys()))
        self.total_functions_in_pool = len(self.dataset_pool)
        
        # --- 核心修正：重新定义数据集的“长度” ---
        if steps_per_epoch is None:
            self.length = len(self.anchor_indices)
            console.log(f"初始化完成。一轮将遍历所有 [bold green]{self.length:,}[/bold green] 个锚点。")
        else:
            self.length = steps_per_epoch
            console.log(f"初始化完成。一轮被定义为 [bold yellow]{self.length:,}[/bold yellow] 个随机采样的步骤。")


    def __len__(self) -> int:
        """返回我们定义的一轮的长度。"""
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        获取一个 (Anchor, Positive, Negative) 三元组的文本。
        注意：输入的idx在这里被忽略，我们总是进行随机采样。
        """
        # --- 核心修正：总是随机采样锚点 ---
        # 这确保了即使在有限的steps_per_epoch内，我们也能看到多样化的数据。
        anchor_idx = random.choice(self.anchor_indices)

        positive_idx = random.choice(self.positive_map[anchor_idx])

        negative_idx = random.randint(0, self.total_functions_in_pool - 1)
        
        positive_set_for_anchor = set(self.positive_map[anchor_idx])
        while negative_idx == anchor_idx or negative_idx in positive_set_for_anchor:
            negative_idx = random.randint(0, self.total_functions_in_pool - 1)

        anchor_text = self.dataset_pool[anchor_idx]['text']
        positive_text = self.dataset_pool[positive_idx]['text']
        negative_text = self.dataset_pool[negative_idx]['text']

        return {
            "anchor": anchor_text,
            "positive": positive_text,
            "negative": negative_text,
        }


@dataclass
class TripletDataCollator:
    """
    一个为对比学习三元组任务设计的数据整理器。
    """
    tokenizer: PreTrainedTokenizer
    max_length: int
    instruction: str = "Represent this LLVM IR for searching for similar functions:"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        anchor_texts = [self.instruction + e["anchor"] for e in examples]
        positive_texts = [self.instruction + e["positive"] for e in examples]
        negative_texts = [self.instruction + e["negative"] for e in examples]

        anchor_inputs = self.tokenizer(
            anchor_texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        positive_inputs = self.tokenizer(
            positive_texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        negative_inputs = self.tokenizer(
            negative_texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )

        batch = {
            "anchor_input_ids": anchor_inputs.input_ids,
            "anchor_attention_mask": anchor_inputs.attention_mask,
            "positive_input_ids": positive_inputs.input_ids,
            "positive_attention_mask": positive_inputs.attention_mask,
            "negative_input_ids": negative_inputs.input_ids,
            "negative_attention_mask": negative_inputs.attention_mask,
        }
        return batch


def debug_dataset_and_collator():
    """
    一个用于调试的函数，测试 OnlineTripletDataset 和 TripletDataCollator 是否正常工作。
    """
    console.rule("[bold yellow]开始调试数据集和数据整理器[/bold yellow]")
    try:
        # --- 使用虚拟配置进行测试 ---
        # !!! 重要: 请将下面的路径替换为您自己脚本生成的真实路径 !!!
        dataset_pool_path = "data/processed_dataset/train_dataset_pool"
        positive_map_path = "data/processed_dataset/train_positive_map.pkl"
        model_name = "Qwen/Qwen2-Embedding-0.5B"
        max_length = 2048

        # --- 1. 测试 OnlineTripletDataset ---
        console.log("[bold]1. 测试 OnlineTripletDataset...[/bold]")
        train_dataset = OnlineTripletDataset(
            dataset_pool_path=dataset_pool_path,
            positive_map_path=positive_map_path,
            steps_per_epoch=100 # 在调试时使用一个小的虚拟epoch长度
        )
        
        # 从数据集中取一个样本进行检查
        sample_item = train_dataset[0]
        console.log("成功从数据集中获取一个样本:")
        console.log(f"  - Anchor text (前50个字符): [green]{sample_item['anchor'][:50]}...[/green]")
        console.log(f"  - Positive text (前50个字符): [green]{sample_item['positive'][:50]}...[/green]")
        console.log(f"  - Negative text (前50个字符): [green]{sample_item['negative'][:50]}...[/green]")
        
        # --- 2. 测试 TripletDataCollator ---
        console.log("\n[bold]2. 测试 TripletDataCollator...[/bold]")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        
        data_collator = TripletDataCollator(
            tokenizer=tokenizer,
            max_length=max_length
        )

        # 准备一个小的批次
        batch_size = 4
        dummy_batch = [train_dataset[i] for i in range(batch_size)]

        # 调用 collator
        processed_batch = data_collator(dummy_batch)
        console.log("成功使用DataCollator处理一个批次的数据。")
        console.log("输出的批次包含以下键:", list(processed_batch.keys()))
        
        # 检查一个张量的形状
        anchor_ids_shape = processed_batch['anchor_input_ids'].shape
        console.log(f"  - 'anchor_input_ids' 的形状: [green]{list(anchor_ids_shape)}[/green]")
        assert list(anchor_ids_shape) == [batch_size, max_length]

        console.rule("[bold green]调试成功！数据集和数据整理器工作正常。[/bold green]")

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
    debug_dataset_and_collator()
