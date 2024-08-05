from typing import List, Optional, Dict, Union
from pydantic import BaseModel, ValidationError
import yaml

class RopeScalingConfig(BaseModel):
    type: Optional[str] = None
    factor: Optional[float] = None

    def dict_without_none(self):
        return {k: v for k, v in self.dict().items() if v is not None}

class OverridesOfModelConfig(BaseModel):
    rope_scaling: Optional[RopeScalingConfig] = None

    def dict_without_none(self):
        return {k: v.dict_without_none() if isinstance(v, RopeScalingConfig) and v is not None else v for k, v in self.dict().items() if v is not None}

class BnbConfigKwargs(BaseModel):
    llm_int8_has_fp16_weight: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def dict_without_none(self):
        return {k: v for k, v in self.dict().items() if v is not None}

class DatasetConfig(BaseModel):
    path: str
    type: str = "alpaca"
    ds_type: Optional[str] = None
    data_files: Optional[List[str]] = None  # Corrected to List[str]
    shards: Optional[int] = None
    name: Optional[str] = None
    train_on_split: str = "train"
    conversation: Optional[str] = None
    field_human: Optional[str] = None
    field_model: Optional[str] = None
    roles: Optional[Dict[str, List[str]]] = None

    def dict_without_none(self):
        return {k: v for k, v in self.dict().items() if v is not None}

class TestDatasetConfig(BaseModel):
    path: str
    ds_type: str = "json"
    split: str = "train"
    type: str = "completion"
    data_files: Optional[List[str]] = None

    def dict_without_none(self):
        return {k: v for k, v in self.dict().items() if v is not None}

class LoftqConfig(BaseModel):
    loftq_bits: Optional[int] = None

    def dict_without_none(self):
        return {k: v for k, v in self.dict().items() if v is not None}

class PeftConfig(BaseModel):
    loftq_config: Optional[LoftqConfig] = None

    def dict_without_none(self):
        return {k: v.dict_without_none() if isinstance(v, LoftqConfig) and v is not None else v for k, v in self.dict().items() if v is not None}

class AxolotlConfig(BaseModel):
    base_model: str = "NousResearch/Llama-2-7b-hf"
    base_model_ignore_patterns: Optional[List[str]] = None
    base_model_config: Optional[str] = None
    revision_of_model: Optional[str] = None
    tokenizer_config: Optional[str] = None
    model_type: str = "LlamaForCausalLM"
    tokenizer_type: str = "LlamaTokenizer"
    trust_remote_code: Optional[bool] = None
    tokenizer_use_fast: bool = True
    tokenizer_legacy: bool = True
    resize_token_embeddings_to_32x: Optional[bool] = None
    is_falcon_derived_model: Optional[bool] = None
    is_llama_derived_model: Optional[bool] = None
    is_qwen_derived_model: Optional[bool] = None
    is_mistral_derived_model: Optional[bool] = None
    overrides_of_model_config: Optional[OverridesOfModelConfig] = None
    bnb_config_kwargs: Optional[BnbConfigKwargs] = None
    gptq: bool = True
    load_in_8bit: bool = True
    load_in_4bit: Optional[bool] = None
    bf16: Union[bool, str] = "auto"
    fp16: Optional[bool] = None
    tf32: bool = False
    bfloat16: bool = True
    float16: bool = True
    gpu_memory_limit: Optional[str] = None
    lora_on_cpu: bool = True
    datasets: List[DatasetConfig] = [DatasetConfig(path="mhenrichsen/alpaca_2k_test")]
    shuffle_merged_datasets: bool = True
    test_datasets: Optional[List[TestDatasetConfig]] = None
    rl: Optional[str] = None
    chat_template: Optional[str] = None
    default_system_message: Optional[str] = None
    dataset_prepared_path: Optional[str] = None
    push_dataset_to_hub: Optional[str] = None
    dataset_processes: Optional[int] = None
    dataset_keep_in_memory: Optional[bool] = None
    hub_model_id: Optional[str] = None
    hub_strategy: Optional[str] = None
    hf_use_auth_token: Optional[bool] = None
    val_set_size: float = 0.05
    dataset_shard_num: Optional[int] = None
    dataset_shard_idx: Optional[int] = None
    sequence_len: int = 4096
    pad_to_sequence_len: bool = True
    sample_packing: bool = True
    eval_sample_packing: Optional[bool] = None
    sample_packing_eff_est: Optional[float] = None
    total_num_tokens: Optional[int] = None
    sample_packing_group_size: Optional[int] = None
    sample_packing_bin_size: Optional[int] = None
    device_map: Optional[str] = None
    max_memory: Optional[str] = None
    adapter: str = "lora"
    lora_model_dir: Optional[str] = None
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_target_linear: bool = True
    peft_layers_to_transform: Optional[List[int]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_fan_in_fan_out: Optional[bool] = None
    loraplus_lr_ratio: Optional[float] = None
    loraplus_lr_embedding: Optional[float] = None
    peft: Optional[PeftConfig] = None
    relora_steps: Optional[int] = None
    relora_warmup_steps: Optional[int] = None
    relora_anneal_steps: Optional[int] = None
    relora_prune_ratio: Optional[float] = None
    relora_cpu_offload: Optional[bool] = None
    wandb_mode: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_watch: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_log_model: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    hf_mlflow_log_artifacts: Optional[bool] = None
    output_dir: str = "./outputs/lora-out"
    torch_compile: Optional[bool] = None
    torch_compile_backend: Optional[str] = None
    gradient_accumulation_steps: int = 4
    micro_batch_size: int = 2
    eval_batch_size: Optional[int] = None
    num_epochs: int = 4
    warmup_steps: int = 10
    warmup_ratio: Optional[float] = None
    learning_rate: float = 0.0002
    lr_quadratic_warmup: Optional[bool] = None
    logging_steps: int = 1
    eval_steps: Optional[Union[int, float]] = None
    evals_per_epoch: int = 4
    save_strategy: Optional[str] = None
    save_steps: Optional[int] = None
    saves_per_epoch: int = 1
    save_total_limit: Optional[int] = None
    max_steps: Optional[int] = None
    eval_table_size: Optional[int] = None
    eval_max_new_tokens: int = 128
    eval_causal_lm_metrics: Optional[List[str]] = None
    loss_watchdog_threshold: Optional[float] = None
    loss_watchdog_patience: Optional[int] = None
    save_safetensors: Optional[bool] = None
    train_on_inputs: bool = False
    group_by_length: bool = False
    gradient_checkpointing: bool = True
    early_stopping_patience: Optional[int] = None
    lr_scheduler: str = "cosine"
    lr_scheduler_kwargs: Optional[Dict[str, Union[str, float]]] = None
    cosine_min_lr_ratio: Optional[float] = None
    cosine_constant_lr_ratio: Optional[float] = None
    lr_div_factor: Optional[float] = None
    optimizer: str = "adamw_bnb_8bit"
    optim_args: Optional[Dict[str, Union[int, float, str]]] = None
    optim_target_modules: Optional[List[str]] = None
    weight_decay: float = 0.0
    adam_beta1: Optional[float] = None
    adam_beta2: Optional[float] = None
    adam_epsilon: Optional[float] = None
    max_grad_norm: Optional[float] = None
    neftune_noise_alpha: Optional[int] = None
    flash_optimum: Optional[bool] = None
    xformers_attention: Optional[bool] = None
    flash_attention: Optional[bool] = None  # Changed to Optional[bool]
    flash_attn_cross_entropy: Optional[bool] = None
    flash_attn_rms_norm: Optional[bool] = None
    flash_attn_fuse_qkv: Optional[bool] = None
    flash_attn_fuse_mlp: Optional[bool] = None
    sdp_attention: Optional[bool] = None
    s2_attention: Optional[bool] = None
    resume_from_checkpoint: Optional[str] = None
    auto_resume_from_checkpoints: Optional[bool] = None
    local_rank: Optional[int] = None
    special_tokens: Optional[Dict[str, str]] = None
    tokens: Optional[List[str]] = None
    fsdp: Optional[str] = None
    fsdp_config: Optional[str] = None
    deepspeed: Optional[str] = None
    ddp_timeout: Optional[int] = None
    ddp_bucket_cap_mb: Optional[int] = None
    ddp_broadcast_buffers: Optional[bool] = None
    torchdistx_path: Optional[str] = None
    pretraining_dataset: Optional[str] = None
    debug: Optional[bool] = None
    seed: Optional[int] = None
    strict: bool = False

    def dict_without_none(self):
        def remove_none(d):
            if isinstance(d, dict):
                return {k: remove_none(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [remove_none(i) for i in d if i is not None]
            else:
                return d
        return remove_none(self.dict())

    def save_to_yaml(self, file_path: str):
        data_dict = self.dict_without_none()
        with open(file_path, 'w') as file:
            yaml.dump(data_dict, file, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, file_path: str):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(**data)

    def set_value(self, key: str, value):
        keys = key.split(".")
        current = self
        for k in keys[:-1]:
            current = getattr(current, k)
        setattr(current, keys[-1], value)

# Example usage
try:
    config = AxolotlConfig()
    config.save_to_yaml('axolotl_config.yaml')
    loaded_config = AxolotlConfig.load_from_yaml('test.yaml')
    loaded_config.set_value('datasets', [DatasetConfig(
        path = "sgsgs",
        type = "shshs"
    )])
    loaded_config.save_to_yaml('axolotl_config_updated.yaml')
except ValidationError as e:
    print(e.json())
