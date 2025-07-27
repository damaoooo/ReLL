import torch
import typer
from rich.console import Console
from unsloth import FastLanguageModel

# 使用Rich打印美观的日志信息
console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_name: str = typer.Argument("Qwen/Qwen3-Embedding-0.6B", help="要测试的Hugging Face模型名称。")
):
    """
    一个专门用于验证Unsloth的快速模型加载和LoRA应用是否被成功启用的脚本。
    """
    console.rule(f"[bold blue]开始验证 Unsloth 加速: {model_name}[/bold blue]")

    # 1. 检查CUDA是否可用
    if not torch.cuda.is_available():
        console.print("[bold red]错误: 未检测到CUDA设备。Unsloth需要一个支持CUDA的GPU。[/bold red]")
        raise typer.Exit(code=1)
    
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        console.print("[bold yellow]警告: 您的GPU算力较低 (sm_80以下)，Unsloth可能无法启用所有优化。[/bold yellow]")
        console.print("Unsloth在Ampere架构 (RTX 30xx, A100) 及更新的GPU上表现最佳。")

    console.log(f"检测到CUDA设备: [green]{torch.cuda.get_device_name(0)}[/green]")

    # 2. 尝试使用Unsloth加载模型
    console.log("\n[bold]1. 正在使用 `Unsloth FastLanguageModel` 加载基础模型...[/bold]")
    console.log("请密切关注下面的日志输出，Unsloth会打印详细的优化信息。")
    console.rule()

    try:
        # Unsloth的核心就是这个加载函数
        # 它会自动处理4-bit量化、数据类型和Flash Attention等所有优化
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            dtype = None, # 自动选择最佳dtype
            load_in_4bit = True,
            trust_remote_code = True,
        )

        console.rule()
        console.print("[bold green]基础模型成功加载！[/bold green]")

        # --- 新增：验证LoRA应用 ---
        console.log("\n[bold]2. 正在使用 `get_peft_model` 应用LoRA适配器...[/bold]")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # LoRA rank
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth", # 关键：使用Unsloth优化的梯度检查点
            random_state = 3407,
        )

        console.print("[bold green]LoRA适配器成功应用！[/bold green]")
        console.log("可训练参数详情:")
        model.print_trainable_parameters()
        
        # 3. 最终确认
        # 检查模型是否已转换为PeftModel
        final_model_class = model.__class__.__name__
        console.log(f"\n最终模型的类名为: [bold magenta]{final_model_class}[/bold magenta]")

        if "PeftModel" in final_model_class:
            console.print("\n[bold green]结论：Unsloth 加速和LoRA配置已成功开启！[/bold green]")
            console.print("模型已准备好进行高效的QLoRA微调。")
        else:
            console.print("\n[bold yellow]结论：Unsloth LoRA未能成功应用。[/bold yellow]")
            console.print("请检查Unsloth的安装和环境兼容性。")

    except Exception as e:
        console.print_exception()
        console.print("\n[bold red]在加载或配置模型时发生错误。[/bold red]")


if __name__ == "__main__":
    app()
