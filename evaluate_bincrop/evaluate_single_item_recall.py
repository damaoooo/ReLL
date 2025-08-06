import logging
import pickle
import random
from pathlib import Path
from typing import List

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from transformers import set_seed

# --- 设置 Rich 和 Typer ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()

# --- GPU加速支持 ---
try:
    import cupy as cp
    GPU_AVAILABLE = True
    console.print("[green]✓ GPU加速已启用 (CuPy)[/green]")
except ImportError:
    GPU_AVAILABLE = False
    console.print("[yellow]⚠ 未安装CuPy，将使用CPU计算[/yellow]")


# --- 参考 evaluate.py 的 recall 计算函数 ---
def process_pools_batch_gpu(all_embeddings, pools_batch, k_values: List[int], use_gpu: bool = True) -> dict:
    """
    处理pools批次，计算与所有嵌入向量的相似度，并返回Recall@K结果。
    使用GPU加速计算相似度。
    """
    recalls = {}
    for k in k_values:
        recalls[k] = [0, 0]  # [成功次数, 总次数]

    anchor_index = [s['anchor'] for s in pools_batch] # [batch_size]
    pool_index = [[s['positive']] + s['negatives'] for s in pools_batch] # [batch_size, pool_size]
    all_embeddings_gpu = cp.asarray(all_embeddings) if use_gpu else all_embeddings
    anchor_emb = all_embeddings_gpu[anchor_index] # [batch_size, embedding_dim]
    pool_emb = all_embeddings_gpu[pool_index] # [batch_size, pool_size, embedding_dim]
    
    anchor_emb = anchor_emb[:, cp.newaxis, :]  # [batch_size, 1, embedding_dim]
    similarities = cp.einsum('bie,bje->bij', anchor_emb, pool_emb)  # [batch_size, 1, pool_size]
    similarities = cp.squeeze(similarities, axis=1)  # [batch_size, pool_size]
    
    # 计算Recall@K
    top_indices = cp.argsort(cp.argsort(-similarities, axis=1), axis=1)[:, 0] + 1
    for k in k_values:
        success, total = 0, 0
        success = (top_indices <= k).sum().item()  # 成功的数量
        total = len(top_indices)
        assert success <= total, "成功次数不能大于总次数"
        recalls[k][0] += success
        recalls[k][1] += total

    return recalls


@app.command()
def main(
    pools_path: Path = typer.Argument(..., help="由evaluate_bincrop.py生成的pools pickle文件路径。", exists=True, file_okay=True),
    embeddings_path: Path = typer.Argument(..., help="预先计算的嵌入向量npy文件路径。", exists=True, file_okay=True),
    ks_str: str = typer.Option("1,5,10,15,20,25,30,35,40,45,50", "--ks", "-k", help="要评估的K值，以逗号分隔。"),
    eval_samples: int = typer.Option(0, "--eval-samples", "-n", help="用于评估的随机pool样本数量。0表示使用所有pools。"),
    seed: int = typer.Option(42, "--seed", "-s", help="用于负采样的随机种子。"),
    use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="是否使用GPU加速计算。"),
    gpu_batch_size: int = typer.Option(512, "--gpu-batch-size", help="GPU批量处理的pool数量。"),
):
    """
    使用evaluate_bincrop.py生成的pools和预先计算的嵌入向量评估二进制函数相似性Recall@K性能。
    """
    console.rule(f"[bold blue]开始二进制函数相似性评估[/bold blue]")
    set_seed(seed)
    
    # GPU可用性检查
    if use_gpu and not GPU_AVAILABLE:
        console.print("[yellow]⚠ 请求使用GPU但CuPy不可用，将回退到CPU计算[/yellow]")
        use_gpu = False
    
    if use_gpu:
        console.print(f"[green]🚀 将使用GPU加速，批量大小: {gpu_batch_size}[/green]")
    else:
        console.print("[blue]💻 使用CPU计算[/blue]")
    
    # --- 1. 加载pools ---
    logging.info(f"正在从 [cyan]{pools_path}[/cyan] 加载pools...")
    with open(pools_path, 'rb') as f:
        pools = pickle.load(f)
    logging.info(f"Pools加载完毕，共 [green]{len(pools)}[/green] 个pools。")
    
    # --- 2. 加载嵌入向量 ---
    logging.info(f"正在从 [cyan]{embeddings_path}[/cyan] 加载嵌入向量...")
    all_embeddings = np.load(embeddings_path)
    logging.info(f"嵌入向量加载完毕，形状为: [green]{all_embeddings.shape}[/green]")
    
    # --- 3. 设置评估参数 ---
    k_values = sorted([int(k.strip()) for k in ks_str.split(',')])
    
    if eval_samples > 0 and eval_samples < len(pools):
        logging.info(f"将从 {len(pools):,} 个样本中随机采样 [yellow]{eval_samples:,}[/yellow] 个进行评估...")
        pools_to_evaluate = random.sample(pools, eval_samples)
    else:
        logging.info(f"将评估所有 {len(pools):,} 个样本...")
        pools_to_evaluate = pools
    
    # --- 4. 批量GPU加速评估 ---
    logging.info("开始批量GPU加速评估...")
    
    temp_results = {k: [0, 0] for k in k_values}
    current_batch_size = gpu_batch_size
    
    for i in track(range(0, len(pools_to_evaluate), current_batch_size), description="正在评估..."):
        pools_batch = pools_to_evaluate[i:i + current_batch_size]

        result = process_pools_batch_gpu(
            all_embeddings,
            pools_batch,  # 只处理当前批量大小
            k_values,
            use_gpu=use_gpu
        )
        # 累加结果
        for k in k_values:
            temp_results[k][0] += result[k][0]
            temp_results[k][1] += result[k][1]
    
    # 将结果转换为百分比
    results = {f"Recall@{k}": temp_results[k][0] / temp_results[k][1] if temp_results[k][1] > 0 else 0 for k in k_values}
    
    # --- 5. 打印结果 ---
    console.rule("[bold green]评估结果[/bold green]")
    table = Table(title="二进制函数相似性 Recall@K 结果")
    table.add_column("指标", justify="left", style="cyan")
    table.add_column("值", justify="right", style="magenta")
    
    for k in k_values:
        table.add_row(f"Recall@{k}", f"{results[f'Recall@{k}']:.4f}")
    
    # 添加统计信息
    table.add_row("", "")  # 空行
    table.add_row("总Pools数", f"{len(pools_to_evaluate):,}")
    if len(pools) > 0:
        table.add_row("Pool大小", f"{len(pools[0]['negatives']) + 1}")  # +1 for positive
    
    console.print(table)


if __name__ == "__main__":
    app()
