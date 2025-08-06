import logging
import pickle
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import typer
from datasets import Dataset, load_from_disk
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table

# --- 设置 Rich 和 Typer ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def build_positive_map(dataset: Dataset) -> Dict[int, List[int]]:
    """
    根据 'file_name' 和 'function_name' 构建正样本对的索引映射。
    这是对您原始逻辑的简化和适配。
    """
    logging.info("正在构建正样本映射...")
    console.log("将数据集关键列转换为Pandas DataFrame以便高效分组...")
    # 只选择需要的列，更节省内存
    df = dataset.select_columns(["file_name", "function_name"]).to_pandas()
    df['original_idx'] = range(len(df))

    # --- 新增：文件名规范化处理 ---
    logging.info("正在规范化文件名以提取源二进制文件名...")
    # 示例: agensgraph-git-bloom.so-Os-3ad76dc09f4f95595f6989901b7f4dd7_functions
    # 1. 移除 '_functions' 后缀
    normalized_names = df['file_name'].str.replace('_functions', '', regex=False)
    # 2. 从右边分割，最多分割两次，分离出主体、优化等级和哈希值
    #    例如 '...so-Os-hash' -> ['...so', 'Os', 'hash']
    parts = normalized_names.str.rsplit('-', n=2)
    # 3. 提取主体部分作为源二进制文件名
    df['source_binary_name'] = parts.str[0]
    
    logging.info(f"文件名规范化完成。示例: '[cyan]{df['file_name'].iloc[0]}[/cyan]' -> '[green]{df['source_binary_name'].iloc[0]}[/green]'", extra={"markup": True})

    logging.info("按 [source_binary_name, function_name] 分组以寻找正样本对...")
    # 使用新创建的 'source_binary_name' 列进行分组
    grouped = df.groupby(['source_binary_name', 'function_name'])

    positive_map = defaultdict(list)
    
    # 使用 rich.progress.track 来显示进度条
    for _, group in track(grouped, description="生成正样本对..."):
        if len(group) > 1:
            indices = group['original_idx'].tolist()
            # 使用itertools.combinations高效生成组内所有配对
            for idx1, idx2 in combinations(indices, 2):
                positive_map[idx1].append(idx2)
                positive_map[idx2].append(idx1)

    logging.info(f"正样本映射构建完成，共找到 [bold green]{len(positive_map):,}[/bold green] 个拥有正样本的函数。", extra={"markup": True})
    return dict(positive_map)


def split_and_save_dataset(
    original_dataset: Dataset,
    positive_map: Dict[int, List[int]],
    output_dir: Path,
    train_ratio: float,
    seed: int,
):
    """
    使用基于图的方法划分数据集，并保存所有必需的文件。
    这是对您之前图划分逻辑的重构和增强。
    """
    console.rule("[bold]数据集划分[/bold]")
    logging.info("构建函数关系图以进行无泄漏划分...")
    
    # 将所有函数索引作为节点添加到图中
    graph = nx.Graph()
    graph.add_nodes_from(range(len(original_dataset)))

    # 根据positive_map添加边
    for anchor_idx, positive_indices in track(positive_map.items(), description="构建关系图..."):
        for positive_idx in positive_indices:
            graph.add_edge(anchor_idx, positive_idx)

    logging.info("寻找图中的连通分量（确保相似函数组不被分割）...")
    # --- 核心修正 ---
    # nx.connected_components 会自动包含所有节点，包括没有连接的孤立节点（它们是大小为1的连通分量）。
    # 不再需要单独处理 nx.isolates()，之前的代码在这里重复添加了孤立节点。
    all_groups = list(nx.connected_components(graph))
    
    logging.info(f"共找到 [bold cyan]{len(all_groups):,}[/bold cyan] 个独立的函数组（包含孤立函数）。", extra={"markup": True})

    # 在“函数组”的层面上进行划分
    random.seed(seed)
    random.shuffle(all_groups)

    num_train_groups = int(len(all_groups) * train_ratio)
    train_groups = all_groups[:num_train_groups]
    validation_groups = all_groups[num_train_groups:]

    train_indices = [idx for group in train_groups for idx in group]
    validation_indices = [idx for group in validation_groups for idx in group]

    # --- 创建并保存数据集池 ---
    logging.info("根据索引创建训练集和验证集池...")
    train_dataset_pool = original_dataset.select(train_indices)
    validation_dataset_pool = original_dataset.select(validation_indices)

    # --- 关键步骤：创建旧索引到新索引的映射 ---
    logging.info("创建旧索引到新索引的映射...")
    train_old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(train_indices)}
    validation_old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(validation_indices)}

    # --- 过滤并翻译 positive_map ---
    def filter_and_translate_map(global_map, old_to_new_map):
        new_map = {}
        # old_to_new_map.keys() 包含了这个子集的所有旧索引
        index_set = set(old_to_new_map.keys())
        for anchor_old, positive_old_list in track(global_map.items(), description="翻译正样本映射..."):
            if anchor_old in index_set:
                # 过滤正样本列表，确保它们也存在于当前数据集中
                filtered_positives_old = [p_idx for p_idx in positive_old_list if p_idx in index_set]
                if filtered_positives_old:
                    # 使用新索引进行翻译
                    anchor_new = old_to_new_map[anchor_old]
                    positives_new = [old_to_new_map[p_idx] for p_idx in filtered_positives_old]
                    new_map[anchor_new] = positives_new
        return new_map

    logging.info("为训练集翻译正样本映射...")
    train_positive_map_new = filter_and_translate_map(positive_map, train_old_to_new_map)
    logging.info("为验证集翻译正样本映射...")
    validation_positive_map_new = filter_and_translate_map(positive_map, validation_old_to_new_map)

    # --- 保存所有产物 ---
    console.rule("[bold]保存产物[/bold]")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_pool_path = output_dir / "train_dataset_pool"
    validation_pool_path = output_dir / "validation_dataset_pool"
    train_map_path = output_dir / "train_positive_map.pkl"
    validation_map_path = output_dir / "validation_positive_map.pkl"

    with console.status("[bold green]正在保存所有文件...[/bold green]", spinner="dots"):
        train_dataset_pool.save_to_disk(str(train_pool_path))
        validation_dataset_pool.save_to_disk(str(validation_pool_path))
        with open(train_map_path, 'wb') as f:
            pickle.dump(train_positive_map_new, f)
        with open(validation_map_path, 'wb') as f:
            pickle.dump(validation_positive_map_new, f)

    # --- 打印最终统计信息 ---
    table = Table(title="[bold]数据集划分结果[/bold]", title_justify="left")
    table.add_column("项目", style="cyan")
    table.add_column("数量", justify="right", style="magenta")
    table.add_row("原始数据集函数总数", f"{len(original_dataset):,}")
    table.add_row("训练集函数数 (Pool)", f"{len(train_dataset_pool):,}")
    table.add_row("验证集函数数 (Pool)", f"{len(validation_dataset_pool):,}")
    table.add_row("训练集有正样本的函数数", f"{len(train_positive_map_new):,}")
    table.add_row("验证集有正样本的函数数", f"{len(validation_positive_map_new):,}")
    table.add_row("训练集数据池保存路径", str(train_pool_path))
    table.add_row("验证集数据池保存路径", str(validation_pool_path))
    table.add_row("训练集映射保存路径", str(train_map_path))
    table.add_row("验证集映射保存路径", str(validation_map_path))
    console.print(table)


@app.command()
def main(
    dataset_path: Path = typer.Argument(..., help="由上一步脚本创建的Hugging Face数据集的路径。", exists=True, file_okay=False, dir_okay=True, readable=True),
    output_dir: Path = typer.Argument(..., help="用于保存训练/验证集产物的输出目录。", file_okay=False, dir_okay=True, writable=True),
    train_ratio: float = typer.Option(0.9, "--ratio", "-r", help="训练集所占的比例。", min=0.1, max=1),
    seed: int = typer.Option(42, "--seed", "-s", help="用于保证划分可复现的随机种子。")
):
    """
    将预处理过的数据集划分为训练集和验证集。

    本脚本采用基于图的方法，确保功能相似的函数组（例如，同一源代码经不同优化编译的版本）
    被完整地划分到同一个数据集中（要么都在训练集，要么都在验证集），从而防止数据泄露。
    """
    console.rule(f"[bold blue]数据集划分脚本 (Train/Validation Split)[/bold blue]")

    # 加载数据集
    logging.info(f"正在从磁盘加载数据集: [cyan]{dataset_path}[/cyan]", extra={"markup": True})
    dataset = load_from_disk(str(dataset_path))

    # 构建正样本映射
    positive_map = build_positive_map(dataset)

    # 执行划分与保存
    split_and_save_dataset(dataset, positive_map, output_dir, train_ratio, seed)
    
    console.rule("[bold green]所有操作完成！[/bold green]")


if __name__ == "__main__":
    app()
