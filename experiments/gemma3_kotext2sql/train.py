import os
import torch

from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


# WSL2 + NCCL 멀티 GPU 안정화
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

accelerator = Accelerator(mixed_precision="no")

# 1. 모델 및 토크나이저 설정
model_id = "google/gemma-3-4b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. config 설정
config = AutoConfig.from_pretrained(model_id)
config.use_cache = False
if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
    config.text_config.use_cache = False

# 3. 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    quantization_config=bnb_config,
    device_map={"": accelerator.local_process_index},
    torch_dtype=torch.float16,
)

# 4. k-bit 학습 준비
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# 5. LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# 6. 데이터셋 로드
dataset = load_dataset("daje/kotext-to-sql-v1", split="train[:10000]")


class Gemma3DataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        if "token_type_ids" in examples[0]:
            token_type_ids = [torch.tensor(example["token_type_ids"]) for example in examples]
            batch["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                token_type_ids,
                batch_first=True,
                padding_value=0,
            )

        return batch


# 7. 텍스트 구성
def build_text(example):
    question = str(example.get("question", "")).strip()
    db_schema = str(example.get("db_schema", "")).strip()
    sql = str(example.get("sql", "")).strip()

    if db_schema:
        user_text = f"{question}\n\n[SCHEMA]\n{db_schema}"
    else:
        user_text = question

    text = (
        f"<start_of_turn>user\n"
        f"{user_text}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{sql}"
        f"<end_of_turn>"
    )
    return text


def tokenize_function(example):
    text = build_text(example)

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    # Gemma3는 학습 시 token_type_ids를 항상 기대한다.
    # 텍스트 전용 샘플은 전부 0으로 두어 일반 causal mask를 사용한다.
    if "token_type_ids" not in tokenized:
        tokenized["token_type_ids"] = [0] * len(tokenized["input_ids"])

    # causal LM labels
    labels = tokenized["input_ids"].copy()
    tokenized["labels"] = labels

    return tokenized


tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
)

# 8. 학습 설정
# 4-bit LoRA 경로는 이미 저정밀도로 계산된다.
# 여기서 Trainer AMP(fp16/bf16)를 추가로 켜면 grad scaler가 dtype 충돌을 일으킬 수 있다.
training_args = SFTConfig(
    output_dir="./artifacts/gemma3_kotext2sql",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=1e-5,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=200,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=True,
    gradient_checkpointing=True,
    report_to="none",
    packing=False,
    remove_unused_columns=False,  # token_type_ids 유지
)

# 9. Trainer 생성
data_collator = Gemma3DataCollator(
    pad_token_id=tokenizer.pad_token_id,
    completion_only_loss=False,
    padding_free=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    data_collator=data_collator,
)

# 10. 학습
trainer.train()

# 11. 저장
save_path = "./artifacts/gemma3_kotext2sql/final_adapter"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
