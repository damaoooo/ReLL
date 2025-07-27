import torch
import typer
from rich.console import Console
from transformers import AutoModel, AutoConfig

# 使用Rich打印美观的日志信息
console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_name: str = typer.Argument("Qwen/Qwen3-Embedding-0.6B", help="要测试的Hugging Face模型名称。")
):
    """
    一个专门用于验证Flash Attention 2是否被成功启用的脚本。
    """
    console.rule(f"[bold blue]开始验证 Flash Attention: {model_name}[/bold blue]")

    # 1. 检查CUDA是否可用
    if not torch.cuda.is_available():
        console.print("[bold red]错误: 未检测到CUDA设备。Flash Attention需要一个支持CUDA的GPU。[/bold red]")
        raise typer.Exit(code=1)
    
    console.log(f"检测到CUDA设备: [green]{torch.cuda.get_device_name(0)}[/green]")

    # 2. 尝试加载模型
    console.log("\n正在尝试使用 `attn_implementation='flash_attention_2'` 加载模型...")
    console.log("请密切关注下面的日志输出，寻找来自 'transformers' 库的明确提示。")
    console.rule()

    try:
        # 加载模型时指定 attn_implementation 和 torch_dtype。
        # 将模型移动到 .to("cuda") 会最终触发实现的选择和日志记录。
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to("cuda")

        console.rule()
        console.print("[bold green]模型成功加载到GPU！[/bold green]")
        
        # 3. 最终确认
        # 检查模型配置，看最终选择的实现是什么
        final_attn_implementation = model.config._attn_implementation
        console.log(f"模型配置中最终确定的注意力实现为: [bold magenta]{final_attn_implementation}[/bold magenta]")

        if final_attn_implementation == "flash_attention_2":
            console.print("\n[bold green]结论：Flash Attention 2 已成功开启！[/bold green]")
        else:
            console.print("\n[bold yellow]结论：Flash Attention 2 未能开启。[/bold yellow]")
            console.print("模型回退到了默认的注意力实现。这通常是由于 `flash-attn` 库的安装或环境兼容性问题。")

    except Exception as e:
        console.print_exception()
        console.print("\n[bold red]在加载模型时发生错误。[/bold red]")


if __name__ == "__main__":
    app()
