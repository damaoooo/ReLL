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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
# --- 核心修正：引入 AutoTokenizer ---
from transformers import AutoTokenizer, set_seed
from datasets import load_from_disk
from numba import njit, prange, types
from numba.typed import Dict as NumbaDict
import torch


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
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    console.print("[green]✓ GPU加速已启用 (PyTorch)[/green]")
else:
    console.print("[yellow]⚠ 未检测到可用GPU，将使用CPU计算[/yellow]")

MRR_CUTOFFS = (10, 30)


@njit
def _seed_numba_rng(seed: int) -> None:
    """为 numba 内部使用的随机数生成器设置种子。"""
    np.random.seed(seed)


def _sanitize_positive_map(positive_map, total_size: int):
    """清洗 positive_map，移除越界/自指向/重复正样本。"""
    sanitized = {}
    stats = {
        "invalid_anchor": 0,
        "invalid_positive": 0,
        "self_positive": 0,
        "duplicate_positive": 0,
        "empty_anchor": 0,
    }

    for raw_anchor_idx, positives in positive_map.items():
        try:
            anchor_idx = int(raw_anchor_idx)
        except (TypeError, ValueError):
            stats["invalid_anchor"] += 1
            continue

        if anchor_idx < 0 or anchor_idx >= total_size:
            stats["invalid_anchor"] += 1
            continue

        seen = set()
        cleaned = []
        for raw_pos_idx in positives:
            try:
                pos_idx = int(raw_pos_idx)
            except (TypeError, ValueError):
                stats["invalid_positive"] += 1
                continue

            if pos_idx < 0 or pos_idx >= total_size:
                stats["invalid_positive"] += 1
                continue
            if pos_idx == anchor_idx:
                stats["self_positive"] += 1
                continue
            if pos_idx in seen:
                stats["duplicate_positive"] += 1
                continue

            seen.add(pos_idx)
            cleaned.append(pos_idx)

        if cleaned:
            sanitized[anchor_idx] = cleaned
        else:
            stats["empty_anchor"] += 1

    return sanitized, stats



# 您的原始函数签名，保持不变
def generate_embeddings_with_tei(dataset, batch_size: int, instruction: str, tei_endpoint: str, tokenizer, max_length: int) -> np.ndarray:
    
    # --- 新增的导入 ---
    import concurrent.futures
    import threading

    # --- 新增: 为并发请求设置一个线程局部session ---
    # 这可以避免多线程环境下requests.Session的潜在问题
    thread_local = threading.local()
    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    # --- 新增: 将循环内的逻辑封装成一个独立的函数 ---
    # 这个函数将由每个线程来执行，负责处理一个批次
    def process_one_batch(batch_texts):
        # 获取当前线程专属的session
        session = get_session()
        
        # --- 下面的代码块与您原来的for循环内部完全相同 ---
        instructed_texts = [instruction + text for text in batch_texts]
        truncated_inputs = tokenizer(
            instructed_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        final_texts_to_send = tokenizer.batch_decode(truncated_inputs['input_ids'], skip_special_tokens=True)
        payload = {"inputs": final_texts_to_send}
        
        try:
            response = session.post(f"{tei_endpoint}/embed", json=payload, timeout=60)
            response.raise_for_status()
            return np.array(response.json(), dtype=np.float32)
        except requests.exceptions.RequestException as e:
            # 当一个请求失败时，打印错误并重新抛出异常
            # executor.map会捕获这个异常，并在主线程中重新引发它
            console.print(f"[bold red]错误: 一个并发请求失败: {e}[/bold red]")
            raise

    # --- 修改: 将原来的for循环替换为ThreadPoolExecutor ---

    # 1. 预先准备好所有的批次数据
    batches = [dataset[i : i + batch_size]['text'] for i in range(0, len(dataset), batch_size)]
    
    # 设置一个合理的并发数，例如8，以确保能充分利用2个GPU
    # 您可以根据需要调整这个值
    MAX_WORKERS = 8 
    all_embeddings = []

    try:
        # 2. 使用并发执行器来处理所有批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # executor.map会自动处理并发，并按顺序返回结果
            results_iterator = executor.map(process_one_batch, batches)
            
            # 创建自定义进度条显示实时速度
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("[cyan]{task.fields[speed]:.1f} 函数/秒"),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    f"并发生成嵌入(Workers: {MAX_WORKERS})",
                    total=len(dataset),
                    speed=0.0
                )
                
                processed_count = 0
                for batch_emb in results_iterator:
                    all_embeddings.append(batch_emb)
                    processed_count += len(batch_emb)
                    # 更新速度：当前已处理数量 / 经过的时间
                    elapsed = progress.tasks[0].elapsed or 0.001  # 避免除零
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    progress.update(task, advance=len(batch_emb), speed=speed)

    except Exception as e:
        # 如果任何一个worker线程中出现异常，程序会在这里中断
        console.print("[bold red]嵌入向量生成过程中发生错误，程序已终止。[/bold red]")
        raise typer.Exit(code=1)

    # 3. 最后一步与原来相同：将所有结果拼接起来
    return np.vstack(all_embeddings)


@njit
def _floyd_sample(pop_size: int, sample_size: int) -> np.ndarray:
    """Floyd算法：在O(k)时间内均匀采样k个不重复整数。"""
    selected = NumbaDict.empty(key_type=types.int64, value_type=types.boolean)
    for j in range(pop_size - sample_size, pop_size):
        t = np.random.randint(0, j + 1)
        if t in selected:
            selected[j] = True
        else:
            selected[t] = True
    out = np.empty(sample_size, dtype=np.int64)
    idx = 0
    for key in selected.keys():
        out[idx] = key
        idx += 1
    return out


@njit
def _map_exclusions(compressed: np.ndarray, exclude_arr: np.ndarray) -> np.ndarray:
    """把压缩空间索引映射回真实索引，保证排除表不被选中。"""
    mapped = compressed.copy()
    while True:
        shift = np.searchsorted(exclude_arr, mapped, side="right")
        new_mapped = compressed + shift
        if np.all(new_mapped == mapped):
            return new_mapped
        mapped = new_mapped


@njit
def _sample_excluding(total_size: int, exclude_arr: np.ndarray, sample_size: int) -> np.ndarray:
    """在排除表之外做均匀无放回采样，返回真实索引。"""
    eligible_size = total_size - exclude_arr.size
    if eligible_size < sample_size:
        return np.empty(0, dtype=np.int64)
    compressed = _floyd_sample(eligible_size, sample_size)
    return _map_exclusions(compressed, exclude_arr)


@njit(parallel=True)
def _build_pools_parallel(anchor_batch: np.ndarray, pos_flat: np.ndarray, pos_offsets: np.ndarray, total_size: int, pool_size: int) -> np.ndarray:
    """并行构建每个锚点的采样池（CPU多核）。"""
    batch_size = anchor_batch.size
    pools = np.full((batch_size, pool_size + 1), -1, dtype=np.int64)
    for i in prange(batch_size):
        anchor_idx = anchor_batch[i]
        start = pos_offsets[anchor_idx]
        end = pos_offsets[anchor_idx + 1]
        pos_len = end - start
        if pos_len <= 0:
            continue
        rand_idx = np.random.randint(start, end)
        positive_anchor_idx = pos_flat[rand_idx]
        exclude_arr = np.empty(pos_len + 1, dtype=np.int64)
        exclude_arr[:pos_len] = pos_flat[start:end]
        exclude_arr[pos_len] = anchor_idx
        exclude_arr.sort()

        mapped = _sample_excluding(total_size, exclude_arr, pool_size)
        if mapped.size != pool_size:
            continue

        pools[i, 0] = positive_anchor_idx
        pools[i, 1:] = mapped

    return pools


def process_anchor_batch_gpu(all_embeddings, anchor_batch, pos_flat, pos_offsets, pool_sizes, k_values: List[int], use_gpu: bool = True):
    """
    处理锚点批次，计算与所有嵌入向量的相似度，并返回Recall@K结果。
    使用GPU加速计算相似度。
    """
    recalls = {}
    for pool_size in pool_sizes:
        recalls[pool_size] = {}
        for k in k_values:
            recalls[pool_size][k] = [0, 0]  # 每次都创建新的列表

    mrr_stats = {cutoff: [0.0, 0] for cutoff in MRR_CUTOFFS}
    mrr_pool_stats = {pool_size: [0.0, 0] for pool_size in pool_sizes}
    max_pool_size = max(pool_sizes)
    pool_size = max_pool_size - 1
    total_size = len(all_embeddings)
    anchor_batch_arr = np.asarray(anchor_batch, dtype=np.int64)
    pools = _build_pools_parallel(anchor_batch_arr, pos_flat, pos_offsets, total_size, pool_size)
    valid_mask = pools[:, 0] >= 0
    if not np.any(valid_mask):
        return recalls, mrr_stats, mrr_pool_stats

    pools = pools[valid_mask]
    anchor_batch_arr = anchor_batch_arr[valid_mask]

    if use_gpu:
        device = all_embeddings.device
        anchor_idx = torch.from_numpy(anchor_batch_arr).to(device=device, dtype=torch.long)
        pool_idx = torch.from_numpy(pools).to(device=device, dtype=torch.long)
        with torch.inference_mode():
            anchors = all_embeddings.index_select(0, anchor_idx)
            embedding_pools = all_embeddings.index_select(0, pool_idx.view(-1)).view(pool_idx.shape[0], pool_idx.shape[1], -1)
            anchor_emb = anchors.unsqueeze(1)
            similarities = torch.bmm(embedding_pools, anchor_emb.transpose(1, 2)).squeeze(-1)
    else:
        anchors = all_embeddings[anchor_batch_arr]
        embedding_pools = all_embeddings[pools]
        anchor_emb = anchors[:, np.newaxis, :]
        similarities = np.matmul(embedding_pools, np.transpose(anchor_emb, (0, 2, 1))).squeeze(-1)

    # 计算MRR（在最大pool中取排名，超过阈值则记为0）
    mrr_sim_slice = similarities[:, :similarities.shape[1]]
    pos_scores = mrr_sim_slice[:, 0:1]
    if use_gpu:
        count_greater = (mrr_sim_slice > pos_scores).sum(dim=1)
        ranks = count_greater + 1
    else:
        count_greater = (mrr_sim_slice > pos_scores).sum(axis=1)
        ranks = count_greater + 1
    for cutoff in MRR_CUTOFFS:
        mrr_cutoff = min(cutoff, mrr_sim_slice.shape[1])
        if use_gpu:
            mrr_scores = torch.where(ranks <= mrr_cutoff, 1.0 / ranks.to(dtype=torch.float32), torch.zeros_like(ranks, dtype=torch.float32))
            mrr_stats[cutoff][0] += float(mrr_scores.sum().item())
            mrr_stats[cutoff][1] += int(mrr_scores.numel())
        else:
            mrr_scores = np.where(
                ranks <= mrr_cutoff,
                1.0 / ranks.astype(np.float32),
                0.0,
            )
            mrr_stats[cutoff][0] += float(mrr_scores.sum())
            mrr_stats[cutoff][1] += int(mrr_scores.size)

    # 计算Recall@K（不做全量排序，直接比较正样本得分排名）
    for pool_size in pool_sizes:
        sim_slice = similarities[:, :pool_size]
        pos_scores = sim_slice[:, 0:1]
        if use_gpu:
            count_greater = (sim_slice > pos_scores).sum(dim=1)
            ranks = count_greater + 1
            mrr_pool_scores = 1.0 / ranks.to(dtype=torch.float32)
            mrr_pool_stats[pool_size][0] += float(mrr_pool_scores.sum().item())
            mrr_pool_stats[pool_size][1] += int(mrr_pool_scores.numel())
        else:
            count_greater = (sim_slice > pos_scores).sum(axis=1)
            ranks = count_greater + 1
            mrr_pool_scores = 1.0 / ranks.astype(np.float32)
            mrr_pool_stats[pool_size][0] += float(mrr_pool_scores.sum())
            mrr_pool_stats[pool_size][1] += int(mrr_pool_scores.size)
        for k in k_values:
            if use_gpu:
                success = (count_greater < k).sum().item()
                total = int(count_greater.numel())
            else:
                success = int((count_greater < k).sum())
                total = int(count_greater.size)
            assert success <= total, f"Success count {success} cannot be greater than total {total}."
            recalls[pool_size][k][0] += success
            recalls[pool_size][k][1] += total

    return recalls, mrr_stats, mrr_pool_stats


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
    _seed_numba_rng(seed)
    
    # GPU可用性检查
    if use_gpu and not GPU_AVAILABLE:
        console.print("[yellow]⚠ 请求使用GPU但PyTorch不可用，将回退到CPU计算[/yellow]")
        use_gpu = False
    
    if use_gpu:
        console.print(f"[green]🚀 将使用GPU加速，批量大小: {gpu_batch_size}[/green]")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        console.print("[blue]💻 使用CPU计算[/blue]")
    
    # --- 1. 加载数据和Tokenizer ---
    logging.info("正在加载数据和Tokenizer...")
    validation_dataset = load_from_disk(str(validation_dataset_pool_path))
    with open(validation_positive_map_path, 'rb') as f:
        positive_map = pickle.load(f)
    positive_map, positive_map_stats = _sanitize_positive_map(positive_map, len(validation_dataset))
    sanitized_count = sum(positive_map_stats.values())
    if sanitized_count > 0:
        logging.warning(
            "positive_map 已清洗: "
            f"invalid_anchor={positive_map_stats['invalid_anchor']}, "
            f"invalid_positive={positive_map_stats['invalid_positive']}, "
            f"self_positive={positive_map_stats['self_positive']}, "
            f"duplicate_positive={positive_map_stats['duplicate_positive']}, "
            f"empty_anchor={positive_map_stats['empty_anchor']}"
        )
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

    if len(all_embeddings) != len(validation_dataset):
        console.print(
            "[bold red]错误: 嵌入向量数量与验证集大小不一致。"
            f" embeddings={len(all_embeddings)}, dataset={len(validation_dataset)}[/bold red]"
        )
        raise typer.Exit(code=1)

    # --- 3. GPU内存预处理 ---
    if use_gpu:
        logging.info("正在将嵌入向量转移到GPU (bfloat16)...")
        all_embeddings_gpu = torch.as_tensor(all_embeddings, dtype=torch.bfloat16, device="cuda")
        logging.info(f"GPU内存使用: {all_embeddings_gpu.numel() * all_embeddings_gpu.element_size() / (1024**3):.2f} GB")
    else:
        all_embeddings_gpu = None


    # --- 4. 设置评估参数 ---
    requested_pool_sizes = sorted(set([2**i for i in range(1, 14)] + [100, 10000]))
    try:
        k_values = sorted({int(k.strip()) for k in ks_str.split(',') if k.strip()})
    except ValueError as exc:
        console.print(f"[bold red]错误: --ks 参数格式不合法: {ks_str}[/bold red]")
        raise typer.Exit(code=1) from exc
    if not k_values or any(k <= 0 for k in k_values):
        console.print(f"[bold red]错误: --ks 必须是正整数列表，当前值为: {ks_str}[/bold red]")
        raise typer.Exit(code=1)
    results = {}

    total_size = len(all_embeddings)
    pos_offsets = np.zeros(total_size + 1, dtype=np.int64)
    pos_flat_list = []
    for idx in range(total_size):
        positives = positive_map.get(idx, [])
        pos_offsets[idx + 1] = pos_offsets[idx] + len(positives)
        pos_flat_list.extend(positives)
    pos_flat = np.asarray(pos_flat_list, dtype=np.int64)
    
    all_possible_anchors = list(positive_map.keys())
    if not all_possible_anchors:
        console.print("[bold red]错误: 清洗后的 positive_map 中没有可评估的锚点。[/bold red]")
        raise typer.Exit(code=1)

    if eval_samples > 0 and eval_samples < len(all_possible_anchors):
        logging.info(f"将从 {len(all_possible_anchors):,} 个可能的锚点中随机采样 [yellow]{eval_samples:,}[/yellow] 个进行评估...")
        anchors_to_evaluate = random.sample(all_possible_anchors, eval_samples)
    else:
        logging.info(f"将评估所有 {len(all_possible_anchors):,} 个锚点...")
        anchors_to_evaluate = all_possible_anchors

    max_feasible_pool_size = min(total_size - len(positive_map[anchor_idx]) for anchor_idx in anchors_to_evaluate)
    pool_sizes = [pool_size for pool_size in requested_pool_sizes if pool_size <= max_feasible_pool_size]
    dropped_pool_sizes = [pool_size for pool_size in requested_pool_sizes if pool_size > max_feasible_pool_size]
    if dropped_pool_sizes:
        logging.warning(
            "以下 pool size 超出当前评估样本可支持上限，已跳过: "
            f"{dropped_pool_sizes} (max_feasible_pool_size={max_feasible_pool_size})"
        )
    if not pool_sizes:
        console.print(
            "[bold red]错误: 当前评估样本不足以构建任意有效检索池。"
            f" max_feasible_pool_size={max_feasible_pool_size}[/bold red]"
        )
        raise typer.Exit(code=1)


    # --- 5. 对不同的池大小进行评估 ---
    logging.info("开始对不同池大小进行批量GPU加速评估...")
    
    temp_results = {}
    for pool_size in pool_sizes:
        temp_results[pool_size] = {k: [0, 0] for k in k_values}
    total_mrr = {cutoff: [0.0, 0] for cutoff in MRR_CUTOFFS}
    total_mrr_by_pool = {pool_size: [0.0, 0] for pool_size in pool_sizes}
    
    
    # 使用自定义进度条显示评估速度
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[speed]:.1f} 锚点/秒"),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "正在评估...",
            total=len(anchors_to_evaluate),
            speed=0.0
        )
        
        processed_anchors = 0
        for i in range(0, len(anchors_to_evaluate), gpu_batch_size):
            anchor_batch = anchors_to_evaluate[i:i + gpu_batch_size]
            result, batch_mrr, batch_mrr_by_pool = process_anchor_batch_gpu(
                all_embeddings_gpu if use_gpu else all_embeddings,
                anchor_batch,
                pos_flat,
                pos_offsets,
                pool_sizes,
                k_values,
                use_gpu=use_gpu
            )
            # 累加结果
            for pool_size in pool_sizes:
                for k in k_values:
                    temp_results[pool_size][k][0] += result[pool_size][k][0]
                    temp_results[pool_size][k][1] += result[pool_size][k][1]
            for cutoff in MRR_CUTOFFS:
                total_mrr[cutoff][0] += batch_mrr[cutoff][0]
                total_mrr[cutoff][1] += batch_mrr[cutoff][1]
            for pool_size in pool_sizes:
                total_mrr_by_pool[pool_size][0] += batch_mrr_by_pool[pool_size][0]
                total_mrr_by_pool[pool_size][1] += batch_mrr_by_pool[pool_size][1]
            
            # 更新进度和速度
            processed_anchors += len(anchor_batch)
            elapsed = progress.tasks[0].elapsed or 0.001
            speed = processed_anchors / elapsed if elapsed > 0 else 0
            progress.update(task, advance=len(anchor_batch), speed=speed)
                
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
    for cutoff in MRR_CUTOFFS:
        mrr_value = total_mrr[cutoff][0] / total_mrr[cutoff][1] if total_mrr[cutoff][1] > 0 else 0
        console.print(f"[bold green]MRR@{cutoff}: {mrr_value:.4f}[/bold green]")

    mrr_pool_table = Table(title="MRR@P 在不同大小检索池中的表现")
    mrr_pool_table.add_column("Pool Size", justify="right", style="cyan")
    mrr_pool_table.add_column("MRR@P", justify="right", style="green")
    for pool_size in pool_sizes:
        mrr_p = total_mrr_by_pool[pool_size][0] / total_mrr_by_pool[pool_size][1] if total_mrr_by_pool[pool_size][1] > 0 else 0
        mrr_pool_table.add_row(f"{pool_size:,}", f"{mrr_p:.4f}")
    console.print(mrr_pool_table)


if __name__ == "__main__":
    app()
