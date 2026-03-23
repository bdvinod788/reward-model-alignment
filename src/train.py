import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

parser = argparse.ArgumentParser(description = 'Run Training')
parser.add_argument('--model', type=str, default='google/gemma-2b-it')
parser.add_argument('--dataset', type=str, default='hh')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--output', type=str, default=os.path.join(os.environ.get("RM_PROJECT", "."), "checkpoint"))

args = parser.parse_args()

dataset = load_dataset('Anthropic/hh-rlhf', split='train')

model = AutoModelForSequenceClassification.from_pretrained(    
    args.model,
    num_labels = 1,
    torch_dtype = torch.bfloat16,
    device_map = "auto"
)

tokenizer = AutoTokenizer.from_pretrained(args.model)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def format_dataset(sample):
    tokenized_chosen = tokenizer(sample['chosen'], truncation=True, max_length=512)
    tokenized_rejected = tokenizer(sample['rejected'], truncation=True, max_length=512)

    return {
        "input_ids_chosen": tokenized_chosen['input_ids'],
        "attention_mask_chosen": tokenized_chosen['attention_mask'],
        "input_ids_rejected": tokenized_rejected['input_ids'],
        "attention_mask_rejected": tokenized_rejected['attention_mask']
    }
    
formatted_dataset = dataset.map(format_dataset, batched = False)

formatted_dataset = formatted_dataset.train_test_split(test_size = 0.1)
train_dataset = formatted_dataset['train']
test_dataset = formatted_dataset['test']

config = RewardConfig(
    output_dir = args.output,
    num_train_epochs = args.epochs,
    per_device_train_batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    bf16 = True,
    max_length = 512,
    logging_steps = 50,
    eval_strategy = "steps",
    eval_steps = 500,
    save_steps = 500,
    report_to = 'wandb',
)

trainer = RewardTrainer(
    model = model,
    args = config,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    processing_class = tokenizer
)

trainer.train()
