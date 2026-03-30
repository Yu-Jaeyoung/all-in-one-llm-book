import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, TaskType
from huggingface_hub import login


def maybe_hf_login() -> None:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        return

    if not str(token).startswith("hf_"):
        logging.warning("Ignoring Hugging Face token from env because it does not look like an access token.")
        return

    login(token=token, add_to_git_credential=False, skip_if_logged_in=True)


def warn_if_full_precision_may_oom(script_args) -> None:
    if script_args.use_qlora or not torch.cuda.is_available():
        return

    for device_index in range(torch.cuda.device_count()):
        total_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        if total_memory_gb <= 12:
            logging.warning(
                "GPU %s has %.1f GiB VRAM. Full-precision LoRA can OOM with %s on this setup; "
                "prefer use_qlora=true or a smaller model.",
                device_index,
                total_memory_gb,
                script_args.model_name,
            )


def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False

    # PyTorch can report BF16 support via emulation on pre-Ampere GPUs, but AMP gradient unscale
    # still fails there. Use a strict hardware capability gate instead.
    return all(torch.cuda.get_device_capability(index)[0] >= 8 for index in range(torch.cuda.device_count()))


def enforce_supported_mixed_precision(training_args) -> None:
    if not getattr(training_args, "bf16", False):
        return

    if supports_bf16():
        return

    logging.warning("BF16 is not supported on this GPU. Falling back to FP16.")
    training_args.bf16 = False
    training_args.fp16 = True


def cast_trainable_params_to_fp32_if_needed(model, training_args) -> None:
    if getattr(training_args, "bf16", False) or supports_bf16():
        return

    converted_param_names = []
    for param_name, param in model.named_parameters():
        if not param.requires_grad or param.dtype != torch.bfloat16:
            continue
        param.data = param.data.to(torch.float32)
        converted_param_names.append(param_name)

    if converted_param_names:
        preview = ", ".join(converted_param_names[:8])
        if len(converted_param_names) > 8:
            preview += f", ... (+{len(converted_param_names) - 8} more)"
        logging.warning(
            "Converted %d trainable BF16 parameters to FP32 for unsupported GPU: %s",
            len(converted_param_names),
            preview,
        )


def log_trainable_param_dtypes(model, stage: str) -> None:
    dtype_counts = {}
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + 1

    logging.info("[%s] trainable parameter dtypes: %s", stage, dtype_counts)


def log_cuda_memory_snapshot(stage: str) -> None:
    if not torch.cuda.is_available():
        logging.info("[%s] CUDA is not available.", stage)
        return

    logging.info("[%s] CUDA memory snapshot", stage)
    for device_index in range(torch.cuda.device_count()):
        allocated_mb = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
        total_mb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 2)
        free_mb = total_mb - reserved_mb
        logging.info(
            "GPU %s | allocated=%.0fMB reserved=%.0fMB free~=%.0fMB total=%.0fMB",
            device_index,
            allocated_mb,
            reserved_mb,
            free_mb,
            total_mb,
        )


def log_model_device_map(model) -> None:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is None:
        logging.info("hf_device_map is not set on the model.")
        return

    logging.info("hf_device_map: %s", hf_device_map)

    modules_by_device = {}
    for module_name, device in hf_device_map.items():
        modules_by_device.setdefault(str(device), []).append(module_name)

    for device, module_names in modules_by_device.items():
        preview = ", ".join(module_names[:8])
        if len(module_names) > 8:
            preview += f", ... (+{len(module_names) - 8} more)"
        logging.info("device=%s | modules=%d | %s", device, len(module_names), preview)


maybe_hf_login()

# ── 데이터셋 준비 ──────────────────────────────────────────────
dataset = load_dataset("beomi/KoAlpaca-v1.1a")
columns_to_remove = list(dataset["train"].features)

system_prompt = (
    "당신은 다양한 분야의 전문가들이 제공한 지식과 정보를 바탕으로 만들어진 AI 어시스턴트입니다. "
    "사용자들의 질문에 대해 정확하고 유용한 답변을 제공하는 것이 당신의 주요 목표입니다."
)

train_dataset = dataset.map(
    lambda sample: {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    },
)
train_dataset = train_dataset.map(remove_columns=columns_to_remove, batched=False)
train_dataset = train_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
train_dataset["test"].to_json("test_dataset.json",  orient="records", force_ascii=False)


# ── ScriptArguments ────────────────────────────────────────────
@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={"help": "데이터셋 파일 경로"},
    )
    model_name: str = field(
        default=None,
        metadata={"help": "SFT 학습에 사용할 모델 ID"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "SFT Trainer에 사용할 최대 시퀀스 길이"},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "True 이면 QLoRA(INT4), False 이면 LoRA(FP16)"},
    )


# ── 학습 함수 ──────────────────────────────────────────────────
def training_function(script_args, training_args):
    warn_if_full_precision_may_oom(script_args)

    # 데이터 로딩
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "test_dataset.json"),
        split="train",
    )

    # ── Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def template_dataset(examples):
        return {
            "text": tokenizer.apply_chat_template(
                examples["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset  = test_dataset.map(template_dataset,  remove_columns=["messages"])

    with training_args.main_process_first(desc="샘플 로그 출력"):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # ── LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # r=8 ,
        r=4 ,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "v_proj",
        ],
    )

    # ── 모델 로딩
    if script_args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # Leave runtime headroom on GPU 1 for activations and temporary buffers.
        max_memory = {1: "5000MB", 0: "10000MB",  "cpu": "32GB"}
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory=max_memory,
            dtype=torch.float16,
            use_cache=False,
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            attn_implementation="sdpa",
            dtype=torch.float16,
            # device_map 제거 ← accelerate가 담당
            # max_memory={0: "10500MB", 1: "7000MB"},
            use_cache=False if training_args.gradient_checkpointing else True,
        )

    log_model_device_map(model)
    log_cuda_memory_snapshot("after_model_load")

    if training_args.gradient_checkpointing and not script_args.use_qlora:
        model.gradient_checkpointing_enable()

    training_args.dataset_text_field = "text"
    training_args.max_length = script_args.max_seq_length
    # training_args.packing = True
    training_args.packing = False
    training_args.dataset_kwargs = {
        "add_special_tokens": False,
        "append_concat_token": False,
    }

    # ── Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    cast_trainable_params_to_fp32_if_needed(trainer.model, training_args)
    log_trainable_param_dtypes(trainer.model, "after_trainer_init")
    log_cuda_memory_snapshot("after_trainer_init")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    log_trainable_param_dtypes(trainer.model, "before_train")
    log_cuda_memory_snapshot("before_train")
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model()
    print(f"모델 저장 완료: {training_args.output_dir}")


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()
    enforce_supported_mixed_precision(training_args)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    set_seed(training_args.seed)
    training_function(script_args, training_args)
