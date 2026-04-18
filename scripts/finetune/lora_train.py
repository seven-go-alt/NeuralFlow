"""
NeuralFlow Fine-tuning Module (LoRA/QLoRA)
展示基于 Hugging Face Transformers 与 PEFT 的微调能力。
用于优化 Agent 在特定领域（如金融、医疗或特定 API 调用）的表现。
"""

import os
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

def train_lora(
    model_name_or_path: str = "Qwen/Qwen2-7B-Instruct",
    dataset_path: str = "data/sft_data.jsonl",
    output_dir: str = "output/neuralflow-lora",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
):
    # 1. 配置 4-bit 量化 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 准备模型进行量化训练
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. 配置 LoRA (JD 要求掌握的核心点)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 针对 Llama/Qwen 结构
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. 加载与处理数据集
    # 假设数据格式为 {"instruction": "...", "input": "...", "output": "..."}
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def tokenize_function(examples):
        # 实现对话模版拼接逻辑
        instructions = [f"Human: {ins}\nAI: " for ins in examples["instruction"]]
        inputs = tokenizer(instructions, truncation=True, padding="max_length", max_length=512)
        targets = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 6. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True, # 如果硬件支持
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none" # 可改为 tensorboard 或 wandb
    )

    # 7. 启动训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )

    print("--- Starting NeuralFlow Finetuning ---")
    trainer.train()
    
    # 8. 保存适配器
    model.save_pretrained(os.path.join(output_dir, "adapter"))
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    # 仅作为演示，实际运行需要 GPU 环境和数据集
    # train_lora()
    print("Fine-tuning script ready. Run with GPU and valid dataset.")
