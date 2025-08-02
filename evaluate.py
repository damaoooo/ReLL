import logging
import pickle
import random
from pathlib import Path
from typing import List

import numpy as np
import requests
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
# --- 核心修正：引入 AutoTokenizer ---
from transformers import AutoTokenizer, set_seed
from datasets import load_from_disk


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


def generate_embeddings_with_tei(dataset, batch_size: int, instruction: str, tei_endpoint: str, tokenizer, max_length: int) -> np.ndarray:
    """
    使用Text Embedding Inference (TEI)服务器为整个数据集生成嵌入向量。
    在发送前，先在客户端进行精确的、与训练时一致的截断。
    """
    all_embeddings = []
    session = requests.Session()

    for i in track(range(0, len(dataset), batch_size), description="正在通过TEI生成嵌入向量..."):
        batch_texts = dataset[i : i + batch_size]['text']
        
        # --- 核心修正：在客户端进行精确截断 ---
        # 1. 首先将指令和函数文本拼接，完全模拟模型在训练时看到的输入
        instructed_texts = [instruction + text for text in batch_texts]

        # 2. 使用Tokenizer对拼接后的完整文本进行截断
        truncated_inputs = tokenizer(
            instructed_texts,
            truncation=True,
            max_length=max_length, # 使用我们指定的长度，例如2048
            padding=False, # 我们不需要padding，只需要截断
        )
        # 3. 将截断后的token IDs解码回文本
        final_texts_to_send = tokenizer.batch_decode(truncated_inputs['input_ids'], skip_special_tokens=True)
        
        # 4. 发送给TEI，此时不再需要TEI进行截断
        payload = {"inputs": final_texts_to_send}
        
        try:
            response = session.post(f"{tei_endpoint}/embed", json=payload, timeout=60)
            response.raise_for_status()
            
            batch_embeddings = np.array(response.json(), dtype=np.float32)
            all_embeddings.append(batch_embeddings)
            
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]错误: TEI请求失败 at batch {i}-{i+batch_size}: {e}[/bold red]")
            console.print("请确保您的TEI服务器正在运行，并且endpoint地址正确。")
            raise typer.Exit(code=1)

    return np.vstack(all_embeddings)


def process_anchor_batch_gpu(all_embeddings, anchor_batch, positive_map, pool_sizes, k_values: List[int], use_gpu: bool = True) -> dict:
    """
    处理锚点批次，计算与所有嵌入向量的相似度，并返回Recall@K结果。
    使用GPU加速计算相似度。
    """
    recalls = {}
    for pool_size in pool_sizes:
        recalls[pool_size] = {}
        for k in k_values:
            recalls[pool_size][k] = [0, 0]  # 每次都创建新的列表
    
    anchors = all_embeddings[anchor_batch]
    max_pool_size = max(pool_sizes)
    pool_size = max_pool_size - 1
    pools = []
    batch_size = len(anchor_batch)
    
    for i in range(batch_size):
        anchor_idx = anchor_batch[i]
        positive = positive_map[anchor_idx]
        positive_anchor_idx = random.choice(positive)
        
        while True:
            # Sample the random Pool
            candidate_indices = np.random.choice(len(all_embeddings), size=pool_size, replace=False)
            candidate_indices_set = set(candidate_indices)
            if positive_anchor_idx in candidate_indices_set or anchor_idx in candidate_indices_set:
                continue
            else:
                batch_pool = np.concatenate(([positive_anchor_idx], candidate_indices))
                pools.append(batch_pool)
                break

    pools = np.array(pools)
    embedding_pools = all_embeddings[pools] # size: (batch_size, pool_size, embedding_dim)
    anchor_emb = anchors[:, np.newaxis, :]  # size: (batch_size, 1, embedding_dim)
    

    anchor_emb_gpu = cp.asarray(anchor_emb)
    embedding_pools_gpu = cp.asarray(embedding_pools)
    similarities = cp.einsum('bij,bkj->bik', anchor_emb_gpu, embedding_pools_gpu)
    similarities = cp.squeeze(similarities, axis=1)  # size: (batch_size, pool_size)


        
    # 计算Recall@K
    for pool_size in pool_sizes:
        top_indices = cp.argsort(cp.argsort(-similarities[:, :pool_size], axis=1), axis=1)[:, 0] + 1
        for k in k_values:
            success, total = 0, 0
            success = (top_indices <= k).sum()
            total = len(top_indices)
            assert success <= total, f"Success count {success} cannot be greater than total {total}."
            recalls[pool_size][k][0] += success
            recalls[pool_size][k][1] += total

    return recalls


@app.command()
def main(
    validation_dataset_pool_path: Path = typer.Argument(..., help="验证集数据池的路径。", exists=True, dir_okay=True),
    validation_positive_map_path: Path = typer.Argument(..., help="验证集正样本映射.pkl文件路径。", exists=True, file_okay=True),
    tei_endpoint: str = typer.Option("http://gpu1.damaoooo.com:8080", help="Text Embedding Inference (TEI) 服务器的URL。"),
    ks_str: str = typer.Option("1,5,10,15,20,25,30,35,40,45,50", "--ks", "-k", help="要评估的K值，以逗号分隔。"),
    batch_size: int = typer.Option(128, "--batch-size", "-b", help="发送到TEI服务器的批量大小。"),
    max_length: int = typer.Option(2048, "--max-length", help="发送到TEI前，将'指令+文本'整体截断到的最大token长度。"),
    eval_samples: int = typer.Option(187256, "--eval-samples", "-n", help="用于评估的随机锚点样本数量。"),
    embeddings_path: Path = typer.Option(None, "--embeddings-path", "-e", help="用于保存/加载嵌入向量Numpy文件的路径。"),
    seed: int = typer.Option(42, "--seed", "-s", help="用于负采样的随机种子。"),
    use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="是否使用GPU加速计算。"),
    gpu_batch_size: int = typer.Option(512, "--gpu-batch-size", help="GPU批量处理的锚点数量。"),
):
    """
    在验证集上评估模型的函数检索性能 (Recall@K)，使用TEI服务器加速嵌入生成，GPU加速相似度计算。
    """
    console.rule(f"[bold blue]开始使用TEI进行模型评估[/bold blue]")
    set_seed(seed)
    
    # GPU可用性检查
    if use_gpu and not GPU_AVAILABLE:
        console.print("[yellow]⚠ 请求使用GPU但CuPy不可用，将回退到CPU计算[/yellow]")
        use_gpu = False
    
    if use_gpu:
        console.print(f"[green]🚀 将使用GPU加速，批量大小: {gpu_batch_size}[/green]")
    else:
        console.print("[blue]💻 使用CPU计算[/blue]")
    
    # --- 1. 加载数据和Tokenizer ---
    logging.info("正在加载数据和Tokenizer...")
    validation_dataset = load_from_disk(str(validation_dataset_pool_path))
    with open(validation_positive_map_path, 'rb') as f:
        positive_map = pickle.load(f)
    # 加载Tokenizer用于客户端截断
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)


    # --- 2. 生成或加载所有嵌入向量 ---
    if embeddings_path and embeddings_path.exists():
        logging.info(f"正在从 [cyan]{embeddings_path}[/cyan] 加载已缓存的嵌入向量...")
        all_embeddings = np.load(embeddings_path)
        logging.info(f"嵌入向量加载完毕，形状为: [green]{all_embeddings.shape}[/green]")
    else:
        instruction = "Represent this LLVM IR for searching for similar functions:"
        all_embeddings = generate_embeddings_with_tei(validation_dataset, batch_size, instruction, tei_endpoint, tokenizer, max_length)
        logging.info(f"嵌入向量生成完毕，形状为: [green]{all_embeddings.shape}[/green]")
        
        if embeddings_path:
            logging.info(f"正在将新生成的嵌入向量缓存到 [cyan]{embeddings_path}[/cyan]...")
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, all_embeddings)
            logging.info("缓存完成。")

    # --- 3. GPU内存预处理 ---
    if use_gpu:
        logging.info("正在将嵌入向量转移到GPU...")
        all_embeddings_gpu = cp.asarray(all_embeddings)
        logging.info(f"GPU内存使用: {all_embeddings_gpu.nbytes / (1024**3):.2f} GB")
    else:
        all_embeddings_gpu = None


    # --- 4. 设置评估参数 ---
    pool_sizes = [2**i for i in range(1, 14)] + [100, 10000]
    # Sort it
    pool_sizes = sorted(pool_sizes)
    k_values = sorted([int(k.strip()) for k in ks_str.split(',')])
    max_k = max(k_values)
    results = {}
    
    all_possible_anchors = list(positive_map.keys())
    if eval_samples > 0 and eval_samples < len(all_possible_anchors):
        logging.info(f"将从 {len(all_possible_anchors):,} 个可能的锚点中随机采样 [yellow]{eval_samples:,}[/yellow] 个进行评估...")
        anchors_to_evaluate = random.sample(all_possible_anchors, eval_samples)
    else:
        logging.info(f"将评估所有 {len(all_possible_anchors):,} 个锚点...")
        anchors_to_evaluate = all_possible_anchors


    # --- 5. 对不同的池大小进行评估 ---
    logging.info("开始对不同池大小进行批量GPU加速评估...")
    
    temp_results = {}
    for pool_size in pool_sizes:
        temp_results[pool_size] = {k: [0, 0] for k in k_values}
    
    
    for i in track(range(0, len(anchors_to_evaluate), gpu_batch_size), description="正在评估..."):
        anchor_batch = anchors_to_evaluate[i:i + gpu_batch_size]
        result = process_anchor_batch_gpu(
            all_embeddings_gpu if use_gpu else all_embeddings,
            anchor_batch,
            positive_map,
            pool_sizes,
            k_values,
            use_gpu=use_gpu
        )
        # 累加结果
        for pool_size in pool_sizes:
            for k in k_values:
                temp_results[pool_size][k][0] += result[pool_size][k][0]
                temp_results[pool_size][k][1] += result[pool_size][k][1]
                
    # 将结果转换为百分比
    
    for pool_size in pool_sizes:
        results[pool_size] = {f"Recall@{k}": temp_results[pool_size][k][0] / temp_results[pool_size][k][1] if temp_results[pool_size][k][1] > 0 else 0 for k in k_values}

    # --- 6. 打印结果 ---
    console.rule("[bold green]评估结果[/bold green]")
    table = Table(title="Recall@K 在不同大小的检索池中的表现")
    table.add_column("Pool Size", justify="right", style="cyan")
    for k in k_values:
        table.add_column(f"Recall@{k}", justify="right", style="magenta")

    for pool_size, recalls in results.items():
        row_data = [f"{pool_size:,}"] + [f"{recalls[f'Recall@{k}']:.4f}" for k in k_values]
        table.add_row(*row_data)
        
    console.print(table)


if __name__ == "__main__":
    app()
