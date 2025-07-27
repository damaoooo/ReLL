import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import typer
from datasets import Dataset, Features, Value, load_dataset
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from transformers import AutoTokenizer

# --- 设置 Rich 和 Typer ---
# 使用 Rich 来美化日志输出
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
# Typer 应用实例
app = typer.Typer(pretty_exceptions_show_locals=False)
# Rich Console 实例，用于打印美观的输出
console = Console()


def filter_short_functions(example: dict, min_body_lines: int = 3) -> bool:
    """
    一个用于过滤函数的辅助函数。
    如果函数体内的有效代码行数小于min_body_lines，则返回False。
    """
    text = example.get('text', '')
    if not text:
        return False

    # 寻找函数体的开始和结束位置
    try:
        start_brace = text.index('{')
        end_brace = text.rindex('}')
    except ValueError:
        # 如果没有 '{' 或 '}'，说明它可能是一个外部声明，应该被过滤掉
        return False
    
    # 提取函数体
    body = text[start_brace + 1 : end_brace]
    
    # 计算非空行的数量
    lines = body.strip().split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    return len(non_empty_lines) > min_body_lines


def process_jsonl_dataset(jsonl_path: Path, model_name: str, num_proc: int, force_rerun: bool) -> Dataset:
    """
    从一个包含 'text' 和 'metadata' 字段的 JSONL 文件创建并处理Hugging Face数据集。
    """
    # 如果强制重新运行，则不使用缓存
    use_cache = not force_rerun
    if not use_cache:
        logging.warning("已激活 --force-rerun 选项，将不使用缓存强制重新处理数据。")

    logging.info(f"正在从 JSONL 文件 '[cyan]{jsonl_path}[/cyan]' 加载数据集...", extra={"markup": True})
    raw_dataset = load_dataset('json', data_files=str(jsonl_path), split='train')
    original_count = raw_dataset.num_rows
    logging.info(f"原始数据集加载完成，包含 [bold green]{original_count:,}[/bold green] 行。", extra={"markup": True})

    # --- 新增：过滤短函数 ---
    logging.info("正在过滤掉函数体过短的函数 (例如，仅包含跳转的函数)...")
    # 使用 .filter() 高效地应用我们的过滤逻辑
    filtered_dataset = raw_dataset.filter(
        filter_short_functions,
        num_proc=num_proc,
        load_from_cache_file=use_cache # <--- 控制缓存
    )
    filtered_count = len(filtered_dataset)
    logging.info(f"过滤完成。移除了 [bold yellow]{original_count - filtered_count:,}[/bold yellow] 个短函数。剩余 [bold green]{filtered_count:,}[/bold green] 个函数。", extra={"markup": True})


    logging.info(f"正在从 '[cyan]{model_name}[/cyan]' 加载Tokenizer...", extra={"markup": True})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logging.info(f"正在使用 [bold yellow]{num_proc}[/bold yellow] 个CPU核心并行处理所有剩余行...", extra={"markup": True})
    console.log("[italic]Hugging Face `datasets` 库将显示其自己的并行处理进度条...[/italic]")

    def tokenize_and_extract_metadata(batch):
        token_lens = [len(tokenizer.encode(text, add_special_tokens=False)) if text and isinstance(text, str) else -1 for text in batch['text']]
        file_names = [meta.get('file_name', '') if meta else '' for meta in batch['metadata']]
        function_names = [meta.get('function_name', '') if meta else '' for meta in batch['metadata']]
        return {"token_len": token_lens, "file_name": file_names, "function_name": function_names}

    processed_dataset = filtered_dataset.map(
        tokenize_and_extract_metadata,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=['metadata'],
        load_from_cache_file=use_cache # <--- 控制缓存
    )

    final_dataset = processed_dataset.filter(lambda example: example['token_len'] != -1)
    logging.info(f"成功处理了 [bold green]{len(final_dataset):,}[/bold green] 行。", extra={"markup": True})

    return final_dataset


def plot_cdf(token_lengths_array, output_path: Path):
    """
    绘制并保存Token长度的累积分布函数（CDF）图。
    """
    logging.info("正在计算并绘制CDF图...")
    token_lengths = token_lengths_array.to_numpy()
    sorted_lengths = np.sort(token_lengths)
    yvals = np.arange(len(sorted_lengths)) / float(len(sorted_lengths) - 1)
    plt.figure(figsize=(12, 7))
    plt.plot(sorted_lengths, yvals)
    plt.title('Token Count CDF in LLVM IR Functions (After Filtering)')
    plt.xlabel(f'Token Count (from Qwen Tokenizer)')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.xscale('linear')
    plt.xlim(0, np.percentile(sorted_lengths, 99.5))
    plt.savefig(output_path, dpi=300)
    logging.info(f"CDF图已保存至 [cyan]{output_path}[/cyan]", extra={"markup": True})


def print_statistics_table(token_lengths_array):
    """
    使用 Rich Table 打印Token长度的统计信息。
    """
    console.log("正在计算统计数据...")
    token_lengths = token_lengths_array.to_numpy()
    table = Table(title="[bold]Token 长度统计信息 (过滤后)[/bold]", title_justify="left")
    table.add_column("统计项", justify="right", style="cyan", no_wrap=True)
    table.add_column("值", justify="left", style="magenta")
    table.add_row("总函数数量", f"{len(token_lengths):,}")
    table.add_row("最小长度", f"{np.min(token_lengths)}")
    table.add_row("最大长度", f"{np.max(token_lengths):,}")
    table.add_row("平均长度", f"{np.mean(token_lengths):.2f}")
    table.add_row("中位数 (50%)", f"{np.median(token_lengths):.0f}")
    table.add_row("95% 分位数", f"{np.percentile(token_lengths, 95):.0f}")
    table.add_row("99% 分位数", f"{np.percentile(token_lengths, 99):.0f}")
    console.print(table)


@app.command()
def main(
    jsonl_path: Path = typer.Argument(..., help="包含数据的JSONL文件路径。", exists=True, file_okay=True, dir_okay=False, readable=True),
    output_dataset_path: Path = typer.Argument(..., help="处理后的Hugging Face数据集的保存路径。", file_okay=False, dir_okay=True, writable=True),
    model_name: str = typer.Option("Qwen/Qwen3-Embedding-0.6B", "--model", "-m", help="用于Tokenization的预训练模型名称。"),
    cdf_plot_path: Path = typer.Option("token_length_cdf_filtered.png", "--plot-path", "-p", help="生成的CDF图的保存路径。", file_okay=True, dir_okay=False, writable=True),
    num_proc: int = typer.Option(32, "--num-proc", "-n", help="用于数据处理的并行CPU核心数。"),
    force_rerun: bool = typer.Option(False, "--force-rerun", help="强制重新处理数据，不使用缓存。")
):
    """
    从一个JSONL文件加载数据，过滤短函数，分析Token长度，并保存为Hugging Face数据集。
    """
    console.rule("[bold blue]数据预处理、过滤与分析脚本[/bold blue]")
    processed_dataset = process_jsonl_dataset(jsonl_path, model_name, num_proc, force_rerun)
    
    token_lengths_array = processed_dataset.data.column('token_len')
    
    print_statistics_table(token_lengths_array)
    plot_cdf(token_lengths_array, cdf_plot_path)
    
    with console.status("[bold green]正在将最终数据集保存到磁盘...[/bold green]", spinner="earth"):
        processed_dataset.save_to_disk(str(output_dataset_path))
    logging.info(f"数据集已成功保存到 [cyan]{output_dataset_path}[/cyan]", extra={"markup": True})
    console.rule("[bold green]处理完成！[/bold green]")


if __name__ == "__main__":
    app()
