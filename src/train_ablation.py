import os
os.environ["WANDB_MODE"] = "offline"

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset, concatenate_datasets


parser = argparse.ArgumentParser(description="Run Training")
parser.add_argument("--model", type=str, default="google/gemma-2b-it")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["hh", "ultrafeedback", "mixed"],
    default="ultrafeedback",
)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(os.environ.get("RM_PROJECT", "."), "checkpoint"),
)
parser.add_argument("--resume", type=str, default=None)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
)
model.config.pad_token_id = tokenizer.eos_token_id


def load_data(dataset_name):
    if dataset_name == "hh":
        return load_dataset("Anthropic/hh-rlhf", split="train")
    elif dataset_name == "ultrafeedback":
        return load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    else:
        hh = load_dataset("Anthropic/hh-rlhf", split="train")
        uf = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        
        def flatten_uf(sample):
            return {
                "chosen": sample["prompt"] + sample["chosen"][-1]["content"],
                "rejected": sample["prompt"] + sample["rejected"][-1]["content"]
            }
        
        uf = uf.map(flatten_uf).select_columns(["chosen", "rejected"])
        
        return concatenate_datasets([hh, uf])
        


def format_dataset(sample):
    if isinstance(sample["chosen"], list):
        chosen_text = sample["prompt"] + sample["chosen"][-1]["content"]
        tokenized_chosen = tokenizer(chosen_text, truncation=True, max_length=512)

        rejected_text = sample["prompt"] + sample["rejected"][-1]["content"]
        tokenized_rejected = tokenizer(rejected_text, truncation=True, max_length=512)

        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }
    else:
        tokenized_chosen = tokenizer(sample["chosen"], truncation=True, max_length=512)
        tokenized_rejected = tokenizer(
            sample["rejected"], truncation=True, max_length=512
        )

        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }


dataset = load_data(args.dataset)

formatted_dataset = dataset.map(format_dataset, batched=False)
formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)

train_dataset = formatted_dataset["train"]
test_dataset = formatted_dataset["test"]

run_name = f"gemma-2b-{args.dataset}-ep{args.epochs}"


config = RewardConfig(
    output_dir=f"{args.output}_{args.dataset}",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    bf16=True,
    max_length=512,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    report_to="none",
    run_name=run_name,
)

trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)


if args.resume:
    import numpy as np
    import torch.serialization

    torch.serialization.add_safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    )
    trainer.train(resume_from_checkpoint=args.resume)
else:
    trainer.train()
