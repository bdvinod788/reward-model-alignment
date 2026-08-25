import os
import json
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

parser = argparse.ArgumentParser(description="RM calibration analysis on RewardBench")
parser.add_argument("--rm_path", type=str, required=True)
parser.add_argument("--dataset", type=str, default="allenai/reward-bench")
parser.add_argument("--split", type=str, default="filtered")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_bins", type=int, default=10)
parser.add_argument("--output", type=str, default="outputs/calibration/results.json")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.rm_path)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.truncation_side = "left"

model = AutoModelForSequenceClassification.from_pretrained(
    args.rm_path, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

dataset = load_dataset(args.dataset, split=args.split)


def to_text(prompt, response):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def score_batch(texts):
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1).float().cpu().numpy()
    return logits


chosen_scores = []
rejected_scores = []

for start in range(0, len(dataset), args.batch_size):
    batch = dataset[start : start + args.batch_size]
    chosen_texts = [to_text(p, c) for p, c in zip(batch["prompt"], batch["chosen"])]
    rejected_texts = [to_text(p, r) for p, r in zip(batch["prompt"], batch["rejected"])]
    chosen_scores.extend(score_batch(chosen_texts).tolist())
    rejected_scores.extend(score_batch(rejected_texts).tolist())
    done = start + len(batch["prompt"])
    if done % (args.batch_size * 20) < args.batch_size:
        print(f"  scored {done}/{len(dataset)}")

chosen_scores = np.array(chosen_scores)
rejected_scores = np.array(rejected_scores)
margins = chosen_scores - rejected_scores
probs = 1 / (1 + np.exp(-margins))  # P(chosen is the preferred response)

confidences = np.maximum(probs, 1 - probs)
correct = (margins > 0).astype(float)

brier = float(np.mean((probs - 1) ** 2))

bin_edges = np.linspace(0.5, 1.0, args.num_bins + 1)
bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, args.num_bins - 1)

n = len(confidences)
ece = 0.0
reliability = []
for b in range(args.num_bins):
    mask = bin_indices == b
    count = int(mask.sum())
    if count == 0:
        reliability.append(
            {
                "bin": b,
                "range": [float(bin_edges[b]), float(bin_edges[b + 1])],
                "count": 0,
                "avg_confidence": None,
                "accuracy": None,
            }
        )
        continue
    avg_conf = float(confidences[mask].mean())
    acc = float(correct[mask].mean())
    ece += (count / n) * abs(avg_conf - acc)
    reliability.append(
        {
            "bin": b,
            "range": [float(bin_edges[b]), float(bin_edges[b + 1])],
            "count": count,
            "avg_confidence": avg_conf,
            "accuracy": acc,
        }
    )

results = {
    "rm_path": args.rm_path,
    "dataset": args.dataset,
    "split": args.split,
    "num_examples": n,
    "overall_accuracy": float(correct.mean()),
    "brier_score": brier,
    "ece": ece,
    "reliability": reliability,
}

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)

print(
    f"\nnum_examples={n}  accuracy={results['overall_accuracy']:.4f}  "
    f"brier={brier:.4f}  ece={ece:.4f}"
)
print(f"{'bin range':<16}{'count':>8}{'avg_conf':>12}{'accuracy':>12}")
for r in reliability:
    if r["count"] == 0:
        continue
    lo, hi = r["range"]
    print(
        f"[{lo:.2f}, {hi:.2f})   {r['count']:>6}  "
        f"{r['avg_confidence']:>10.3f}  {r['accuracy']:>10.3f}"
    )

print(f"\nSaved to {args.output}")
