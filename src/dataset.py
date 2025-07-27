import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    def __init__(self, dataset_pool_path: str, positive_map_path: str):
        """
        初始化数据集。

        Args:
            dataset_pool_path (str): 由 '03_split_dataset.py' 生成的数据集池路径。
            positive_map_path (str): 由 '03_split_dataset.py' 生成的正样本映射.pkl文件路径。
        """
        super().__init__()
        # 加载包含所有函数文本的数据池
        self.dataset_pool = load_from_disk(dataset_pool_path)
        # --- 核心修正 ---
        # console.log() 不接受 'extra' 参数。Markup默认是开启的。
        console.log(f"加载自 [cyan]{dataset_pool_path}[/cyan] 的数据集池，共 {len(self.dataset_pool):,} 个函数。")

        # 加载正样本映射
        with open(positive_map_path, 'rb') as f:
            self.positive_map = pickle.load(f)
        # --- 核心修正 ---
        console.log(f"加载自 [cyan]{positive_map_path}[/cyan] 的正样本映射。")
        
        # 我们的 "dataset" 是所有可以作为锚点的函数，即 positive_map 的键。
        # 对键进行排序可以保证每次运行时的顺序一致性。
        self.anchor_indices = sorted(list(self.positive_map.keys()))
        self.total_functions_in_pool = len(self.dataset_pool)
        
        # --- 核心修正 ---
        console.log(f"初始化完成。可用的锚点数量: [bold green]{len(self.anchor_indices):,}[/bold green]")

    def __len__(self) -> int:
        """返回数据集中可作为锚点的样本数量。"""
        return len(self.anchor_indices)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        根据给定的索引，获取一个 (Anchor, Positive, Negative) 三元组的文本。
        """
        # 1. 选择锚点 (Anchor)
        anchor_idx = self.anchor_indices[idx]

        # 2. 选择正样本 (Positive)
        positive_idx = random.choice(self.positive_map[anchor_idx])

        # 3. 选择负样本 (Negative)
        negative_idx = random.randint(0, self.total_functions_in_pool - 1)
        
        positive_set_for_anchor = set(self.positive_map[anchor_idx])
        while negative_idx == anchor_idx or negative_idx in positive_set_for_anchor:
            negative_idx = random.randint(0, self.total_functions_in_pool - 1)

        # 4. 从数据池中获取这三个样本的 'text' 字段
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
    它接收一批次的文本三元组，并将其转换为模型所需的PyTorch张量。
    """
    tokenizer: PreTrainedTokenizer
    max_length: int
    # --- 新增：为模型添加指令 ---
    instruction: str = "Represent this LLVM IR for searching for similar functions:"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 从批次中分离出文本列表，并在每个文本前加上指令
        anchor_texts = [self.instruction + e["anchor"] for e in examples]
        positive_texts = [self.instruction + e["positive"] for e in examples]
        negative_texts = [self.instruction + e["negative"] for e in examples]

        # 2. 将这三组文本分别进行 tokenize 和 padding。
        #    这是截断(truncation)发生的地方。
        anchor_inputs = self.tokenizer(
            anchor_texts,
            padding="max_length", # 填充到max_length
            truncation=True,      # 超过max_length则截断
            max_length=self.max_length,
            return_tensors="pt"
        )
        positive_inputs = self.tokenizer(
            positive_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        negative_inputs = self.tokenizer(
            negative_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 3. 将所有 tokenized inputs 组合到一个字典中返回
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
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        max_length = 2048

        # --- 1. 测试 OnlineTripletDataset ---
        console.log("[bold]1. 测试 OnlineTripletDataset...[/bold]")
        train_dataset = OnlineTripletDataset(
            dataset_pool_path=dataset_pool_path,
            positive_map_path=positive_map_path
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
