import logging
import os
from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from transformers import set_seed

# 在导入任何Hugging Face库之前，设置这个环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 引入我们项目中的其他模块 ---
from src.dataset import OnlineTripletDataset, TripletDataCollator
from src.model import load_model_and_tokenizer
from src.trainer import TripletTrainer, TripletTrainingArguments

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
    config_path: Path = typer.Argument(..., help="训练配置文件的路径 (YAML格式)。", exists=True, dir_okay=False, readable=True),
    resume_from_checkpoint: bool = typer.Option(False, "--resume", "-r", help="是否从最新的检查点恢复训练。")
):
    """
    启动模型微调训练流程。
    """
    console.rule(f"[bold blue]启动函数相似性模型训练 (Hugging Face PEFT版)[/bold blue]")

    # --- 1. 加载配置文件 ---
    logging.info(f"正在从 [cyan]{config_path}[/cyan] 加载配置...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    lora_config = config['lora']
    data_config = config['data']
    training_config = config['training']
    
    set_seed(training_config.get('seed', 42))

    # --- 2. 加载模型和Tokenizer ---
    # --- 核心回滚：重新传递 quantization_bits 参数 ---
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_config['name'],
        quantization_bits=model_config['quantization_bits'],
        lora_config_dict=lora_config
    )

    # --- 3. 加载数据集和DataCollator ---
    logging.info("正在准备训练和验证数据集...")
    train_dataset = OnlineTripletDataset(
        dataset_pool_path=data_config['train_dataset_pool_path'],
        positive_map_path=data_config['train_positive_map_path']
    )
    validation_dataset = OnlineTripletDataset(
        dataset_pool_path=data_config['validation_dataset_pool_path'],
        positive_map_path=data_config['validation_positive_map_path']
    )
    
    data_collator = TripletDataCollator(
        tokenizer=tokenizer,
        max_length=model_config['max_length'],
        instruction=model_config.get('instruction', '')
    )

    # --- 4. 配置训练参数 ---
    logging.info("正在配置训练参数...")
    training_args = TripletTrainingArguments(**training_config)

    # --- 5. 初始化 Trainer ---
    logging.info("正在初始化 Trainer...")
    trainer = TripletTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )
    
    # --- 6. 开始训练 ---
    console.rule("[bold green]开始训练[/bold green]")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # --- 7. 保存最终的模型和状态 ---
    logging.info("训练完成。正在保存最终的模型适配器...")
    trainer.save_model() 
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    console.rule("[bold green]所有操作完成！[/bold green]")


if __name__ == "__main__":
    app()
