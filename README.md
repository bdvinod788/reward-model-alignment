# Reward model Training for Alignment

Training a Bradley-Terry reward model on human preference data.
Studying reward hacking, overoptimization, and calibration.

## Setup
See 'scripts/setup.sh'

## Structure
- 'src/' - training and evaluation code
- 'configs/' - training configs
- 'scripts/' - SLURM job scripts
- 'notebooks/' - analysis and visualization


## Results

All three variants trained via `src/train_ablation.py` on a unified chat-template
+ left-truncation pipeline (an earlier inconsistent-format pipeline produced
below-chance accuracy on two of three runs).

### RewardBench overall accuracy
- HH-RLHF: 70.4%
- UltraFeedback: 70.1%
- Mixed (HH + UltraFeedback): **72.5%** (best of the three)

### Data ablation findings
- Mixing datasets improved overall accuracy and safety-refusal performance
  rather than degrading it - the mixed model leads on `refusals-dangerous`
  (67%, vs. 55% HH-alone and 8% UF-alone).
- UltraFeedback alone stays weak on refusal-specific categories even on the
  fixed pipeline (`refusals-dangerous` 8%, `xstest-should-refuse` 47%) - likely
  a genuine property of that dataset (no red-team/harmlessness data), not a
  pipeline artifact.

### Pending
- Reward model calibration analysis (ECE, reliability diagrams)
- Best-of-N overoptimization study
- DPO vs explicit RM comparison





