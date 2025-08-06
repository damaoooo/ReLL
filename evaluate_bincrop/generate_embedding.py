import random
import json
import pickle
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Tuple

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from datasets import Dataset, load_from_disk

# 创建rich和typer实例
console = Console()
app = typer.Typer(help="为函数相似性评估创建pools的CLI工具。")


def create_nested_dict():
    """创建嵌套defaultdict的工厂函数，用于multiprocessing"""
    return defaultdict(dict)


def parse_filename(filename: str):
    """
    从文件名中解析出二进制名称和优化级别。
    文件名格式: {binary-name}-{OX}-{hash}_functions
    这个版本可以处理binary-name中包含连字符'-'的情况。
    """
    try:
        # 从右边分割字符串，最多分割两次
        # 例如: "my-binary-name-O0-hash123_functions.c" -> ['my-binary-name', 'O0', 'hash123_functions.c']
        parts = filename.rsplit('-', 2)
        binary_name = parts[0]
        optimization_level = parts[1]
        return binary_name, optimization_level
    except IndexError:
        # 如果分割失败（例如，格式不匹配），则返回None
        return None, None


def process_dataset_chunk(args: Tuple[List, str, str, int]) -> Tuple[Dict, List[int]]:
    """
    处理数据集的一个chunk，返回lookup映射和level2索引
    """
    chunk_data, level1, level2, pool_size = args
    
    # 构建局部lookup映射 - 使用常规dict而不是defaultdict
    local_lookup = {}
    local_level2_indices = []
    
    # 构建lookup映射
    for i, item in chunk_data:
        binary_name, opt_level = parse_filename(item['file_name'])
        if not binary_name:
            continue
        
        if binary_name not in local_lookup:
            local_lookup[binary_name] = {}
        if item['function_name'] not in local_lookup[binary_name]:
            local_lookup[binary_name][item['function_name']] = {}
            
        local_lookup[binary_name][item['function_name']][opt_level] = i
        if opt_level == level2:
            local_level2_indices.append(i)
    
    return local_lookup, local_level2_indices


def build_pools_chunk(args: Tuple[List[int], Dict, List[int], str, str, int]) -> List[Dict]:
    """
    为指定的anchor索引范围构建pools
    """
    anchor_indices, dataset_items, all_level2_indices, level1, level2, pool_size = args
    
    pools = []
    num_negatives_to_sample = pool_size - 1
    
    # 将全局lookup转换为普通dict以提高查找速度
    global_lookup = {}
    for binary_name, funcs in dataset_items['lookup'].items():
        global_lookup[binary_name] = {}
        for func_name, opts in funcs.items():
            global_lookup[binary_name][func_name] = dict(opts)
    
    for i in anchor_indices:
        anchor_item = dataset_items['items'][i]
        binary_name, opt_level = parse_filename(anchor_item['file_name'])
        
        if opt_level != level1:
            continue
            
        function_name = anchor_item['function_name']
        
        if (binary_name in global_lookup and 
            function_name in global_lookup[binary_name] and 
            level2 in global_lookup[binary_name][function_name]):
            positive_index = global_lookup[binary_name][function_name][level2]
        else:
            continue
            
        possible_negatives = [idx for idx in all_level2_indices if idx != positive_index]
        
        if len(possible_negatives) >= num_negatives_to_sample:
            negative_indices = random.sample(possible_negatives, num_negatives_to_sample)
            pools.append({
                'anchor': i,
                'positive': positive_index,
                'negatives': negative_indices
            })
    
    return pools


def create_similarity_pools_parallel(
    dataset: Dataset,
    level1: str,
    level2: str,
    pool_size: int,
    progress: Progress,
    num_processes: int = None
) -> List[Dict]:
    """
    使用多进程的核心逻辑：为函数相似性评估创建pools。
    """
    if pool_size < 2:
        raise ValueError("pool_size必须至少为2 (1 anchor + 1 positive)。")
    
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # 限制最大进程数
    
    console.print(f"使用 [bold yellow]{num_processes}[/bold yellow] 个进程进行并行处理...")
    
    # 将数据集分块
    chunk_size = max(1, len(dataset) // num_processes)
    chunks = []
    
    task1 = progress.add_task("[green]准备数据分块...", total=len(dataset))
    for i in range(0, len(dataset), chunk_size):
        chunk = [(j, dataset[j]) for j in range(i, min(i + chunk_size, len(dataset)))]
        chunks.append((chunk, level1, level2, pool_size))
        progress.update(task1, advance=len(chunk))
    
    # 并行处理数据块
    task2 = progress.add_task("[cyan]并行构建查找映射...", total=len(chunks))
    with Pool(processes=num_processes) as pool:
        results = []
        for result in pool.imap(process_dataset_chunk, chunks):
            results.append(result)
            progress.update(task2, advance=1)
    
    # 合并结果
    global_lookup = defaultdict(create_nested_dict)
    all_level2_indices = []
    
    task3 = progress.add_task("[yellow]合并查找结果...", total=len(results))
    for local_lookup, local_level2_indices in results:
        # 合并lookup
        for binary_name in local_lookup:
            for func_name in local_lookup[binary_name]:
                for opt_level in local_lookup[binary_name][func_name]:
                    global_lookup[binary_name][func_name][opt_level] = local_lookup[binary_name][func_name][opt_level]
        
        # 合并level2索引
        all_level2_indices.extend(local_level2_indices)
        progress.update(task3, advance=1)
    
    console.print(f"查找映射构建完成。找到了 [bold yellow]{len(all_level2_indices)}[/bold yellow] 个 [bold cyan]{level2}[/bold cyan] 级别的函数。")
    
    # 并行构建pools
    # 准备数据用于序列化
    dataset_items = {
        'lookup': dict(global_lookup),
        'items': {i: dataset[i] for i in range(len(dataset))}
    }
    
    # 按索引范围分块处理anchor
    anchor_chunk_size = max(1, len(dataset) // num_processes)
    anchor_chunks = []
    
    task4 = progress.add_task("[magenta]准备Pool构建分块...", total=len(dataset))
    for i in range(0, len(dataset), anchor_chunk_size):
        anchor_indices = list(range(i, min(i + anchor_chunk_size, len(dataset))))
        anchor_chunks.append((anchor_indices, dataset_items, all_level2_indices, level1, level2, pool_size))
        progress.update(task4, advance=len(anchor_indices))
    
    # 并行构建pools
    task5 = progress.add_task("[blue]并行构建评估Pools...", total=len(anchor_chunks))
    with Pool(processes=num_processes) as pool:
        pool_results = []
        for result in pool.imap(build_pools_chunk, anchor_chunks):
            pool_results.append(result)
            progress.update(task5, advance=1)
    
    # 合并所有pools
    all_pools = []
    for pools_chunk in pool_results:
        all_pools.extend(pools_chunk)
    
    return all_pools


@app.command()
def main(
    dataset_path: str = typer.Option("malware-ai/Func-Similarity-60k", help="Hugging Face数据集的路径或名称。"),
    dataset_split: str = typer.Option("train", help="要使用的数据集切片 (例如 'train', 'test')。"),
    level1: str = typer.Option("O0", "--level1", "-l1", help="作为anchor的优化级别。"),
    level2: str = typer.Option("O3", "--level2", "-l2", help="作为positive/negative的优化级别。"),
    pool_size: int = typer.Option(10, "--pool-size", "-p", help="每个pool的大小。1 anchor + 1 positive + (p-2) negatives。"),
    output_file: str = typer.Option("similarity_pools.pkl", "--output", "-o", help="保存结果的pickle文件路径。"),
    max_pools_to_show: int = typer.Option(5, help="在终端中显示的结果示例数量。"),
    num_processes: int = typer.Option(None, "--processes", "-j", help="并行处理的进程数量。默认为CPU核心数。")
):
    """
    从Hugging Face数据集创建函数相似性评估pools，并保存为pickle文件。
    """
    console.print(f"[bold blue]开始处理...[/bold blue]")
    console.print(f"  - [cyan]数据集[/cyan]: {dataset_path} ({dataset_split})")
    console.print(f"  - [cyan]Anchor级别[/cyan]: {level1}")
    console.print(f"  - [cyan]Positive/Negative级别[/cyan]: {level2}")
    console.print(f"  - [cyan]Pool大小[/cyan]: {pool_size}")

    try:
        # --- 加载数据集 ---
        console.print("\n[bold]正在加载数据集...[/bold]")
        dataset = load_from_disk(dataset_path)
        console.print(f"数据集加载成功，包含 [bold yellow]{len(dataset)}[/bold yellow] 条记录。")
    except Exception as e:
        console.print(f"[bold red]错误: 无法加载数据集 '{dataset_path}'.[/bold red]")
        console.print(e)
        raise typer.Exit(code=1)

    # --- 创建进度条并执行核心任务 ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        pools = create_similarity_pools_parallel(dataset, level1, level2, pool_size, progress, num_processes)

    console.print(f"\n[bold green]处理完成！成功创建了 {len(pools)} 个pools。[/bold green]")

    # --- 保存结果 ---
    if pools:
        console.print(f"\n[bold]正在将结果保存到 [underline]{output_file}[/underline]...[/bold]")
        with open(output_file, 'wb') as f:
            pickle.dump(pools, f)
        console.print("保存成功！")

        # --- 打印示例结果 ---
        console.print("\n[bold]结果示例:[/bold]")
        table = Table(title="相似性Pools示例")
        table.add_column("Pool #", style="cyan")
        table.add_column("Anchor Index", style="magenta")
        table.add_column("Anchor Func", style="white")
        table.add_column("Positive Index", style="green")
        table.add_column("Negative Indices", style="yellow")

        for i, pool in enumerate(pools[:max_pools_to_show]):
            anchor_func_name = dataset[pool['anchor']]['function_name']
            table.add_row(
                str(i + 1),
                str(pool['anchor']),
                anchor_func_name,
                str(pool['positive']),
                str(pool['negatives'][:3]) + "..."
            )
        console.print(table)


if __name__ == "__main__":
    app()
