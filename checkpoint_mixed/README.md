---
base_model: google/gemma-2b-it
library_name: transformers
model_name: checkpoint_mixed
tags:
- generated_from_trainer
- trl
- reward-trainer
licence: license
---

# Model Card for checkpoint_mixed

This model is a fine-tuned version of [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with Reward.

### Framework versions

- TRL: 0.19.1
- Transformers: 4.51.0
- Pytorch: 2.5.1+cu121
- Datasets: 4.8.3
- Tokenizers: 0.21.4

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```