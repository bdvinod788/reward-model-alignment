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

### Baseline (HH-RLHF, Gemma-2B-it)
- RewardBench overall: 66.4%
- Strong safety alignment: refusals-offensive 100%, xstest-should-refuse 96.8%
- Weakness: over-refusal on safe-but-sensitive prompts (xstest-should-respond 52.4%)

### Data ablation findings
- UltraFeedback alone: 40.3% overall - safety alignment collapsed completely
  (refusals-dangerous: 62% -> 3%, refusals-offensive: 100% -> 21%)
- Mixed (HH + UltraFeedback): 47.3% overall - safety did not recover despite
  HH-RLHF being present in training data
- Key finding: dataset mixing does not average capabilities - UltraFeedback's
  signal interferes with HH-RLHF's safety alignment

### Pending
- Reward model calibration analysis (ECE, reliability diagrams)
- Best-of-N overoptimization study
- DPO vs explicit RM comparison





