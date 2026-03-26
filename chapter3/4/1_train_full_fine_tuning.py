import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from transformers import AutoConfig, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer, TrlParser
import yaml


DEFAULT_SYSTEM_PROMPT = (
    "당신은 다양한 분야의 전문가들이 제공한 지식과 정보를 바탕으로 만들어진 AI "
    "어시스턴트입니다. 사용자들의 질문에 대해 정확하고 유용한 답변을 제공하는 것이 "
    "당신의 주요 목표입니다. 복잡한 주제에 대해서도 이해하기 쉽게 설명할 수 있으며, "
    "필요한 경우 추가 정보나 관련 예시를 제공할 수 있습니다. 항상 객관적이고 "
    "중립적인 입장을 유지하면서, 최신 정보를 반영하여 답변해 주세요. 사용자의 질문이 "
    "불분명한 경우 추가 설명을 요청하고, 당신이 확실하지 않은 정보에 대해서는 솔직히 "
    "모른다고 말해주세요."
)


@dataclass
class ScriptArguments:
    model_name: str = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "학습에 사용할 Hugging Face 모델 ID"},
    )
    dataset_name: str | None = field(
        default="beomi/KoAlpaca-v1.1a",
        metadata={"help": "허브에서 불러올 데이터셋 이름"},
    )
    dataset_path: str | None = field(
        default=None,
        metadata={
            "help": "로컬 JSON split이 있는 디렉터리. train_dataset.json과 test_dataset.json이 있어야 합니다."
        },
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "dataset_name을 사용할 때 불러올 split 이름"},
    )
    dataset_test_size: float = field(
        default=0.1,
        metadata={"help": "eval_strategy가 활성화된 경우 사용할 검증 데이터 비율"},
    )
    question_key: str = field(
        default="instruction",
        metadata={"help": "질문 또는 instruction 컬럼 이름"},
    )
    input_key: str = field(
        default="input",
        metadata={"help": "추가 입력 컬럼 이름. 비어 있으면 무시합니다."},
    )
    answer_key: str = field(
        default="output",
        metadata={"help": "정답 또는 response 컬럼 이름"},
    )
    system_prompt: str = field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={"help": "원본 instruction 데이터를 chat messages로 바꿀 때 붙일 system prompt"},
    )
    train_file_name: str = field(
        default="train_dataset.json",
        metadata={"help": "dataset_path 아래의 학습용 JSON 파일 이름"},
    )
    eval_file_name: str = field(
        default="test_dataset.json",
        metadata={"help": "dataset_path 아래의 평가용 JSON 파일 이름"},
    )
    preview_samples: int = field(
        default=2,
        metadata={"help": "전처리된 샘플 미리보기 개수"},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "모델 로딩 시 사용할 attention 구현"},
    )


def maybe_hf_login() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


def parse_bool_arg(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def get_cli_arg_value(args: list[str], flag: str) -> tuple[bool, object | None]:
    if flag not in args:
        return False, None

    index = args.index(flag)
    if index + 1 >= len(args) or args[index + 1].startswith("--"):
        return True, True
    return True, args[index + 1]


def set_cli_bool_arg(args: list[str], flag: str, value: bool) -> list[str]:
    rendered = "true" if value else "false"
    if flag in args:
        index = args.index(flag)
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            args[index + 1] = rendered
        else:
            args.insert(index + 1, rendered)
        return args

    args.extend([flag, rendered])
    return args


def load_config_from_args(args: list[str]) -> dict[str, object]:
    if "--config" not in args:
        return {}

    config_index = args.index("--config")
    if config_index + 1 >= len(args):
        return {}

    config_path = Path(args[config_index + 1]).expanduser()
    if not config_path.exists():
        return {}

    with config_path.open(encoding="utf-8") as yaml_file:
        loaded = yaml.safe_load(yaml_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return loaded


def gpu_capabilities() -> list[tuple[int, int]]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())]


def supports_tf32(capabilities: list[tuple[int, int]]) -> bool:
    return bool(capabilities) and all(major >= 8 for major, _ in capabilities)


def supports_bf16(capabilities: list[tuple[int, int]]) -> bool:
    return bool(capabilities) and all(major >= 8 for major, _ in capabilities)


def is_primary_process() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


def prepare_runtime_args(raw_args: list[str]) -> list[str]:
    args = list(raw_args)
    config = load_config_from_args(args)
    capabilities = gpu_capabilities()

    _, cli_tf32_value = get_cli_arg_value(args, "--tf32")
    _, cli_bf16_value = get_cli_arg_value(args, "--bf16")
    fp16_present, cli_fp16_value = get_cli_arg_value(args, "--fp16")

    requested_tf32 = parse_bool_arg(cli_tf32_value)
    if requested_tf32 is None:
        requested_tf32 = parse_bool_arg(config.get("tf32"))

    requested_bf16 = parse_bool_arg(cli_bf16_value)
    if requested_bf16 is None:
        requested_bf16 = parse_bool_arg(config.get("bf16"))

    requested_fp16 = parse_bool_arg(cli_fp16_value) if fp16_present else parse_bool_arg(config.get("fp16"))

    if requested_tf32 and not supports_tf32(capabilities):
        if is_primary_process():
            print("TF32 is not supported on this GPU. Overriding --tf32 to false.", file=sys.stderr)
        args = set_cli_bool_arg(args, "--tf32", False)

    if requested_bf16 and not supports_bf16(capabilities):
        if is_primary_process():
            print("BF16 is not supported on this GPU. Overriding --bf16 to false.", file=sys.stderr)
        args = set_cli_bool_arg(args, "--bf16", False)
        if torch.cuda.is_available() and requested_fp16 is None:
            if is_primary_process():
                print("Enabling FP16 fallback because BF16 is unavailable on this GPU.", file=sys.stderr)
            args = set_cli_bool_arg(args, "--fp16", True)

    return args


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"", "none", "nan"}:
        return ""
    return text


def build_messages(example: dict[str, object], script_args: ScriptArguments) -> dict[str, list[dict[str, str]]]:
    question = clean_text(example.get(script_args.question_key))
    answer = clean_text(example.get(script_args.answer_key))
    extra_input = clean_text(example.get(script_args.input_key))

    if not question:
        raise ValueError(f"Empty question detected for key '{script_args.question_key}'.")
    if not answer:
        raise ValueError(f"Empty answer detected for key '{script_args.answer_key}'.")

    user_content = question
    if extra_input:
        user_content = f"{question}\n\n입력:\n{extra_input}"

    messages = []
    if script_args.system_prompt.strip():
        messages.append({"role": "system", "content": script_args.system_prompt.strip()})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": answer})
    return {"messages": messages}


def ensure_supported_dataset(dataset: Dataset, script_args: ScriptArguments) -> Dataset:
    sample = next(iter(dataset))

    if any(key in sample for key in ("messages", "text", "input_ids")):
        return dataset

    missing_keys = [
        key for key in (script_args.question_key, script_args.answer_key) if key not in sample
    ]
    if missing_keys:
        raise ValueError(
            "Dataset format is not supported. "
            f"Missing keys: {missing_keys}. Available keys: {sorted(sample.keys())}"
        )

    return dataset.map(
        lambda example: build_messages(example, script_args),
        remove_columns=dataset.column_names,
        desc="Convert dataset to chat messages",
    )


def load_local_datasets(
    script_args: ScriptArguments,
    need_eval: bool,
) -> tuple[Dataset, Dataset | None]:
    dataset_dir = Path(script_args.dataset_path).expanduser()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_dir}")

    train_file = dataset_dir / script_args.train_file_name
    eval_file = dataset_dir / script_args.eval_file_name

    if not train_file.exists():
        raise FileNotFoundError(f"Training split not found: {train_file}")
    if need_eval and not eval_file.exists():
        raise FileNotFoundError(f"Evaluation split not found: {eval_file}")

    train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    eval_dataset = None
    if need_eval:
        eval_dataset = load_dataset("json", data_files=str(eval_file), split="train")

    train_dataset = ensure_supported_dataset(train_dataset, script_args)
    if eval_dataset is not None:
        eval_dataset = ensure_supported_dataset(eval_dataset, script_args)

    return train_dataset, eval_dataset


def load_hub_datasets(
    script_args: ScriptArguments,
    training_args: SFTConfig,
    need_eval: bool,
) -> tuple[Dataset, Dataset | None]:
    if script_args.dataset_name is None:
        raise ValueError("dataset_name must be set when dataset_path is not provided.")

    with training_args.main_process_first(desc="Load and preprocess dataset"):
        dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_split)
        dataset = ensure_supported_dataset(dataset, script_args)

    if not need_eval:
        return dataset, None

    if not 0.0 < script_args.dataset_test_size < 1.0:
        raise ValueError("dataset_test_size must be between 0 and 1 when evaluation is enabled.")

    split_dataset = dataset.train_test_split(
        test_size=script_args.dataset_test_size,
        seed=training_args.seed,
    )
    return split_dataset["train"], split_dataset["test"]


def load_datasets(
    script_args: ScriptArguments,
    training_args: SFTConfig,
) -> tuple[Dataset, Dataset | None]:
    need_eval = training_args.eval_strategy != "no"

    if script_args.dataset_path is not None:
        return load_local_datasets(script_args, need_eval)

    return load_hub_datasets(script_args, training_args, need_eval)


def configure_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define either pad_token or eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def validate_model_access(model_name: str) -> None:
    try:
        AutoConfig.from_pretrained(model_name)
    except Exception as error:
        message = str(error)
        gated_repo_markers = (
            "gated repo",
            "403 client error",
            "cannot access gated repo",
            "authorized list",
        )
        if any(marker in message.lower() for marker in gated_repo_markers):
            raise RuntimeError(
                "Cannot access the configured model on Hugging Face Hub.\n"
                f"- model_name: {model_name}\n"
                "- The account behind HF_TOKEN/HUGGINGFACE_HUB_TOKEN (or your cached hf auth login) "
                "does not have permission for this gated repo.\n"
                "- Request access on the model page, or change model_name to a repo you can access."
            ) from error
        raise


def ensure_chat_template_if_needed(tokenizer: AutoTokenizer, dataset: Dataset, training_args: SFTConfig) -> None:
    sample = next(iter(dataset))
    uses_chat_messages = any(key in sample for key in ("messages", "prompt"))

    if uses_chat_messages and tokenizer.chat_template is None and training_args.chat_template_path is None:
        raise ValueError(
            "Conversational datasets require a tokenizer chat template. "
            "Set chat_template_path in the config or use a model tokenizer that already defines one."
        )


def log_sample_previews(dataset: Dataset, tokenizer: AutoTokenizer, preview_samples: int) -> None:
    if preview_samples <= 0 or len(dataset) == 0:
        return

    sample_count = min(preview_samples, len(dataset))
    for index in random.sample(range(len(dataset)), sample_count):
        sample = dataset[index]
        if "messages" in sample:
            preview_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        elif "text" in sample:
            preview_text = sample["text"]
        else:
            preview_text = str({key: sample[key] for key in sorted(sample.keys())})

        print(f"[sample {index}]\n{preview_text}\n")


def build_model_init_kwargs(training_args: SFTConfig, script_args: ScriptArguments) -> dict[str, object]:
    model_init_kwargs = dict(training_args.model_init_kwargs or {})
    model_init_kwargs.setdefault("attn_implementation", script_args.attn_implementation)

    uses_activation_checkpointing = bool(
        training_args.fsdp_config and training_args.fsdp_config.get("activation_checkpointing")
    )
    if training_args.gradient_checkpointing or uses_activation_checkpointing:
        model_init_kwargs["use_cache"] = False
    else:
        model_init_kwargs.setdefault("use_cache", True)

    if "torch_dtype" not in model_init_kwargs:
        if training_args.bf16:
            model_init_kwargs["torch_dtype"] = torch.bfloat16
        elif training_args.fp16:
            model_init_kwargs["torch_dtype"] = torch.float16

    return model_init_kwargs


def training_function(script_args: ScriptArguments, training_args: SFTConfig) -> None:
    maybe_hf_login()

    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    training_args.model_init_kwargs = build_model_init_kwargs(training_args, script_args)

    with training_args.main_process_first(desc="Validate model access"):
        if is_primary_process():
            validate_model_access(script_args.model_name)

    tokenizer = configure_tokenizer(script_args.model_name)
    train_dataset, eval_dataset = load_datasets(script_args, training_args)
    ensure_chat_template_if_needed(tokenizer, train_dataset, training_args)

    with training_args.main_process_first(desc="Log processed samples"):
        log_sample_previews(train_dataset, tokenizer, script_args.preview_samples)

    trainer = SFTTrainer(
        model=script_args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    runtime_args = prepare_runtime_args(sys.argv[1:])
    script_args, training_args = parser.parse_args_and_config(args=runtime_args)

    set_seed(training_args.seed)
    training_function(script_args, training_args)
