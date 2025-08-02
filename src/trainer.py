from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from transformers import Trainer, TrainingArguments
from rich.console import Console

# --- 引入我们项目中的其他模块 ---
from .dataset import OnlineTripletDataset, TripletDataCollator
from .model import load_model_and_tokenizer


# 使用Rich打印美观的日志信息
console = Console()


class TripletTrainer(Trainer):
    """
    一个自定义的Trainer，用于计算Triplet Loss。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = TripletMarginLoss(margin=self.args.triplet_margin)

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        官方推荐的Qwen Embedding模型池化方法。
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss 方法来计算 Triplet Loss。
        """
        # --- 核心回滚：AutoModel的输出不包含 .hidden_states，而是直接提供 .last_hidden_state ---
        anchor_outputs = model(
            input_ids=inputs["anchor_input_ids"],
            attention_mask=inputs["anchor_attention_mask"],
        )
        positive_outputs = model(
            input_ids=inputs["positive_input_ids"],
            attention_mask=inputs["positive_attention_mask"],
        )
        negative_outputs = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
        )

        anchor_emb = self.last_token_pool(anchor_outputs.last_hidden_state, inputs["anchor_attention_mask"])
        positive_emb = self.last_token_pool(positive_outputs.last_hidden_state, inputs["positive_attention_mask"])
        negative_emb = self.last_token_pool(negative_outputs.last_hidden_state, inputs["negative_attention_mask"])

        eps = 1e-8
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1, eps=eps)
        positive_emb = F.normalize(positive_emb, p=2, dim=1, eps=eps)
        negative_emb = F.normalize(negative_emb, p=2, dim=1, eps=eps)

        loss = self.loss_fct(anchor_emb, positive_emb, negative_emb)

        return (loss, {"loss": loss}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool = True, ignore_keys=None):
        """
        重写 prediction_step 方法来处理 evaluation 阶段的 triplet 数据格式。
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # 使用与 compute_loss 相同的逻辑计算损失
            loss = self.compute_loss(model, inputs)
            
            if prediction_loss_only:
                return (loss, None, None)
            
            # 为了兼容评估逻辑，我们需要返回一些 logits
            # 这里我们返回 triplet embeddings 作为 "predictions"
            anchor_outputs = model(
                input_ids=inputs["anchor_input_ids"],
                attention_mask=inputs["anchor_attention_mask"],
            )
            positive_outputs = model(
                input_ids=inputs["positive_input_ids"],
                attention_mask=inputs["positive_attention_mask"],
            )
            negative_outputs = model(
                input_ids=inputs["negative_input_ids"],
                attention_mask=inputs["negative_attention_mask"],
            )

            anchor_emb = self.last_token_pool(anchor_outputs.last_hidden_state, inputs["anchor_attention_mask"])
            positive_emb = self.last_token_pool(positive_outputs.last_hidden_state, inputs["positive_attention_mask"])
            negative_emb = self.last_token_pool(negative_outputs.last_hidden_state, inputs["negative_attention_mask"])

            # 归一化embeddings
            eps = 1e-8
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1, eps=eps)
            positive_emb = F.normalize(positive_emb, p=2, dim=1, eps=eps)
            negative_emb = F.normalize(negative_emb, p=2, dim=1, eps=eps)
            
            # 计算anchor与positive和negative之间的相似度（余弦相似度）
            # 由于embeddings已经归一化，点积就是余弦相似度
            anchor_pos_sim = torch.sum(anchor_emb * positive_emb, dim=1, keepdim=True)  # [batch_size, 1]
            anchor_neg_sim = torch.sum(anchor_emb * negative_emb, dim=1, keepdim=True)  # [batch_size, 1]
            
            # 将相似度拼接作为 predictions: [anchor_pos_sim, anchor_neg_sim]
            # 这样可以直接分析模型是否学会了让anchor与positive更相似，与negative更不相似
            predictions = torch.cat([anchor_pos_sim, anchor_neg_sim], dim=1)  # [batch_size, 2]
            
            # 由于这是无监督学习，我们没有真实的标签，返回 None
            labels = None
            
        return (loss, predictions, labels)

@dataclass
class TripletTrainingArguments(TrainingArguments):
    """
    为Triplet Loss训练添加自定义参数。
    """
    triplet_margin: float = 1.0


def debug_trainer():
    """
    一个用于调试的函数，测试整个训练流程（模型、数据、训练器）是否能协同工作。
    """
    console.rule("[bold yellow]开始调试 Trainer 流程 (Hugging Face PEFT版)[/bold yellow]")
    try:
        # --- 使用虚拟配置进行测试 ---
        dataset_pool_path = "data/processed_dataset/train_dataset_pool"
        positive_map_path = "data/processed_dataset/train_positive_map.pkl"
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        max_length = 2048
        debug_batch_size = 1
        
        # --- 1. 加载模型和Tokenizer ---
        console.log("[bold]1. 加载模型和Tokenizer...[/bold]")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            quantization_bits=4,
            lora_config_dict={
                "r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "bias": "none"
            }
        )

        # --- 2. 加载数据集和DataCollator ---
        console.log("\n[bold]2. 加载数据集和DataCollator...[/bold]")
        train_dataset = OnlineTripletDataset(
            dataset_pool_path=dataset_pool_path,
            positive_map_path=positive_map_path
        )
        data_collator = TripletDataCollator(
            tokenizer=tokenizer,
            max_length=max_length
        )

        # --- 3. 准备一个批次的数据 ---
        console.log("\n[bold]3. 准备一个虚拟批次...[/bold]")
        dummy_batch = data_collator([train_dataset[i] for i in range(debug_batch_size)])
        dummy_batch = {k: v.to(model.device) for k, v in dummy_batch.items()}
        console.log(f"虚拟批次创建成功 (大小={debug_batch_size}) 并已移动到GPU。")

        # --- 4. 测试 compute_loss ---
        console.log("\n[bold]4. 测试 compute_loss 方法...[/bold]")
        training_args = TripletTrainingArguments(
            output_dir="./tmp_debug_trainer",
            triplet_margin=1.0,
            gradient_checkpointing=True,
            label_names=[],
            remove_unused_columns=False
        )
        
        trainer = TripletTrainer(model=model, args=training_args)
        
        loss = trainer.compute_loss(model, dummy_batch)
        console.log(f"成功计算损失！ Loss: [bold green]{loss.item()}[/bold green]")
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "Loss必须是一个标量张量"

        # --- 5. 测试 prediction_step (evaluation逻辑) ---
        console.log("\n[bold]5. 测试 prediction_step 方法 (evaluation逻辑)...[/bold]")
        
        # 测试 prediction_loss_only=True 的情况
        eval_loss, eval_predictions, eval_labels = trainer.prediction_step(
            model, dummy_batch, prediction_loss_only=True
        )
        console.log(f"prediction_loss_only=True: Loss = [bold green]{eval_loss.item()}[/bold green]")
        assert eval_predictions is None and eval_labels is None, "prediction_loss_only=True时应该返回None"
        
        # 测试 prediction_loss_only=False 的情况
        eval_loss, eval_predictions, eval_labels = trainer.prediction_step(
            model, dummy_batch, prediction_loss_only=False
        )
        console.log(f"prediction_loss_only=False: Loss = [bold green]{eval_loss.item()}[/bold green]")
        console.log(f"  - Predictions shape: [bold cyan]{list(eval_predictions.shape)}[/bold cyan]")
        console.log(f"  - Anchor-Positive 相似度: [bold yellow]{eval_predictions[0, 0].item():.4f}[/bold yellow]")
        console.log(f"  - Anchor-Negative 相似度: [bold yellow]{eval_predictions[0, 1].item():.4f}[/bold yellow]")
        
        # 验证predictions的形状和内容
        assert eval_predictions.shape == (debug_batch_size, 2), f"Predictions形状应该是({debug_batch_size}, 2)"
        assert eval_labels is None, "labels应该是None（无监督学习）"
        
        # 验证相似度值在合理范围内（-1到1之间）
        pos_sim = eval_predictions[0, 0].item()
        neg_sim = eval_predictions[0, 1].item()
        assert -1 <= pos_sim <= 1, f"Positive相似度应该在[-1,1]范围内，但得到{pos_sim}"
        assert -1 <= neg_sim <= 1, f"Negative相似度应该在[-1,1]范围内，但得到{neg_sim}"
        
        console.log("[bold green]✓ prediction_step 测试通过！[/bold green]")

        console.rule("[bold green]调试成功！Trainer 训练和评估逻辑都工作正常。[/bold green]")

    except FileNotFoundError:
        console.print(f"[bold red]错误: 找不到数据文件。[/bold red]")
        console.print(f"请确保以下路径是正确的，并且您已经运行了 '03_split_dataset.py' 脚本:")
        console.print(f"  - 数据集池: '{dataset_pool_path}'")
        console.print(f"  - 正样本映射: '{positive_map_path}'")
    except Exception as e:
        console.print_exception()
        console.rule("[bold red]调试失败！请检查上面的错误信息。[/bold red]")


if __name__ == "__main__":
    # 当直接运行此脚本时，执行调试函数
    debug_trainer()
