import logging
from pathlib import Path

import torch
import typer
from peft import PeftModel
from rich.console import Console
from rich.logging import RichHandler
from transformers import AutoModel, AutoTokenizer

# --- 设置 Rich 和 Typer ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
    base_model_name: str = typer.Option("Qwen/Qwen3-Embedding-0.6B", help="基础模型的Hugging Face名称。"),
    lora_adapter_path: Path = typer.Argument(..., help="训练好的LoRA适配器检查点路径。", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(..., help="合并后模型的保存路径。", file_okay=False, dir_okay=True, writable=True),
):
    """
    将训练好的LoRA适配器与基础模型合并，并导出为可供TEI使用的格式。
    """
    console.rule(f"[bold blue]开始合并LoRA适配器: {lora_adapter_path}[/bold blue]")

    # --- 1. 加载基础模型和Tokenizer ---
    logging.info(f"正在从 [cyan]{base_model_name}[/cyan] 加载基础模型...", extra={"markup": True})
    # 关键：为了合并权重，必须加载原始精度的模型，不能进行量化。
    # 我们使用 bfloat16 以匹配训练时的数据类型。
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    logging.info("基础模型和Tokenizer加载完成。")

    # --- 2. 加载并应用LoRA适配器 ---
    logging.info(f"正在从 [cyan]{lora_adapter_path}[/cyan] 加载LoRA适配器...", extra={"markup": True})
    # PeftModel会自动将适配器加载到基础模型上
    model = PeftModel.from_pretrained(model, str(lora_adapter_path))
    logging.info("LoRA适配器加载完成。")

    # --- 3. 合并权重并卸载适配器 ---
    logging.info("正在将LoRA权重合并到基础模型中...")
    # 这是最核心的步骤
    model = model.merge_and_unload()
    logging.info("权重合并完成。")

    # --- 4. 保存最终的模型和Tokenizer ---
    console.rule(f"[bold green]正在保存最终模型到: {output_path}[/bold green]")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with console.status("[bold green]正在保存...[/bold green]", spinner="dots"):
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    
    logging.info("最终模型和Tokenizer已成功保存。")
    console.print(f"\n[bold]下一步:[/bold] 您现在可以将 [cyan]'{output_path}'[/cyan] 目录挂载到您的Text Embedding Inference (TEI) Docker容器中进行部署。")


if __name__ == "__main__":
    app()
