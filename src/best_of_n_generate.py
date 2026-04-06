import os
import json
import torch
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Run Best of N Generation")
parser.add_argument("--model", type=str, default="google/gemma-2b-it")
parser.add_argument("--rm_path", type=str)
parser.add_argument("--n_values", type=int, nargs="+", default=[1, 4, 16, 64])
parser.add_argument("--num_prompts", type=int, default=200)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(
        os.environ.get("RM_PROJECT", "."), "outputs/best_of_n/results.json"
    ),
)

args = parser.parse_args()

# Generator model and tokenizer
gen_tokenizer = AutoTokenizer.from_pretrained(args.model)
gen_tokenizer.pad_token = gen_tokenizer.eos_token

gen_model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto"
)
gen_model.config.pad_token_id = gen_tokenizer.eos_token_id

# Reward model and tokenizer
rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_path)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    args.rm_path, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
)

# Dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="test").select(
    range(args.num_prompts)
)


def extract_prompt(sample):
    return sample["chosen"].rsplit("Assistant:", 1)[0] + "Assistant:"


prompts = [extract_prompt(sample) for sample in dataset]


def generate_responses(prompt, n):
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}
    output_tokens = gen_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=n,
        pad_token_id=gen_tokenizer.eos_token_id,
    )
    return [
        gen_tokenizer.decode(output_tokens[i], skip_special_tokens=True)
        for i in range(n)
    ]


def score_response(prompt, response):
    text = prompt + response
    inputs = rm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(rm_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = rm_model(**inputs)
    return output.logits.squeeze().item()


# Main loop
results = {}
for n in args.n_values:
    print(f"Running N={n}...")
    result_n = []
    for i, prompt in enumerate(prompts):
        max_score = -float("inf")
        best = {}
        responses = generate_responses(prompt, n)
        for response in responses:
            score = score_response(prompt, response)
            if score > max_score:
                max_score = score
                best = {
                    "prompt": prompt,
                    "response": response,
                    "rm_score": score,
                }
        result_n.append(best)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts done")
    results[n] = result_n
    print(
        f"N={n} complete. Avg RM score: {sum(r['rm_score'] for r in result_n) / len(result_n):.4f}"
    )

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {args.output}")
