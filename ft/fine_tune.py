from peft import get_peft_model
import datasets
import mlflow
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
import bitsandbytes as bnb
import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"

"""
Fine Tuner facade.

Pulled from:
https://github.com/cloudera/CML_AMP_Finetune_Foundation_Model_Multiple_Tasks
"""

def get_unique_cache_dir():
    # Use a cache dir specific to this session, since workers and sessions will share project files
    return "~/.cache/" + os.environ['CDSW_ENGINE_ID'] + "/huggingface/datasets"

class AMPFineTuner:
    # Load basemodel from huggingface
    # Default: bigscience/bloom-1b1
    def __init__(self, base_model, auth_token="", ft_job_uuid="", bnb_config=BitsAndBytesConfig()):
        mlflow.set_experiment(ft_job_uuid)

        # Load the base model and tokenizer
        print("Load the base model and tokenizer...\n")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=auth_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        compute_dtype = getattr(torch, "float16")

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map='auto',
            token=auth_token,
        )

        # transformers.TrainingArguments defaults
        self.training_args = TrainingArguments(
            output_dir=f"outputs/{ft_job_uuid}",
            num_train_epochs=1,
            optim="paged_adamw_32bit",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            lr_scheduler_type="constant",
            disable_tqdm=True,
            evaluation_strategy="epoch",
            eval_steps=1,
            save_strategy="epoch",
            report_to='mlflow',
        )

    # Use PEFT library to set LoRA training config and get trainable peft model
    def set_lora_config(self, lora_config):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

        self.lora_config = lora_config
        self.model = get_peft_model(self.model, self.lora_config)

    # Train/Fine-tune model with SFTTrainer and a provided dataset
    def train(
            self,
            train_dataset: datasets.Dataset = None,
            eval_dataset: datasets.Dataset = None,
            dataset_text_field: str = None,
            output_dir: str = None,
            packing=True,
            max_seq_length=1024,
            callbacks: list = None):

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=self.lora_config,
            tokenizer=self.tokenizer,
            dataset_text_field=dataset_text_field,
            packing=packing,
            max_seq_length=max_seq_length,
            args=self.training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=callbacks
        )

        print("Begin Training....")
        trainer.train()
        print("Training Complete!")

        # Save the model and the tokenizer.
        trainer.save_model(output_dir)
        