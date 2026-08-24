import glob
import json
import os

DATASETS = ['hh', 'ultrafeedback', 'mixed']

RESULT_DIRS = {
    'hh':            'outputs/rewardbench_hh',
    'ultrafeedback': 'outputs/rewardbench_ultrafeedback',
    'mixed':         'outputs/rewardbench_mixed',
}

EXAMPLE_COUNTS = {
    'alpacaeval-easy': 100, 'alpacaeval-length': 95, 'alpacaeval-hard': 95,
    'mt-bench-easy': 28, 'mt-bench-med': 40, 'mt-bench-hard': 37,
    'math-prm': 447, 'refusals-dangerous': 100, 'refusals-offensive': 100,
    'llmbar-natural': 100, 'llmbar-adver-neighbor': 134, 'llmbar-adver-GPTInst': 92,
    'llmbar-adver-GPTOut': 47, 'llmbar-adver-manual': 46,
    'xstest-should-refuse': 154, 'xstest-should-respond': 250, 'donotanswer': 136,
    'hep-cpp': 164, 'hep-go': 164, 'hep-java': 164,
    'hep-js': 164, 'hep-python': 164, 'hep-rust': 164,
}

SECTIONS = {
    'Chat': ['alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard',
             'mt-bench-easy', 'mt-bench-med'],
    'Chat Hard': ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor',
                  'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
    'Safety': ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse',
               'xstest-should-respond', 'donotanswer'],
    'Reasoning': ['math-prm', 'hep-cpp', 'hep-go', 'hep-java',
                  'hep-js', 'hep-python', 'hep-rust'],
}

CHANCE = 0.5


def load_latest(result_dir):
    paths = glob.glob(os.path.join(result_dir, '**', '*.json'), recursive=True)
    if not paths:
        raise FileNotFoundError(
            f"No results in {result_dir} - run: sbatch --export=DATASET=... scripts/eval.job"
        )
    return max(paths, key=os.path.getmtime)


def weighted(scores, subsets):
    num = sum(scores[k] * EXAMPLE_COUNTS[k] for k in subsets if k in scores)
    den = sum(EXAMPLE_COUNTS[k] for k in subsets if k in scores)
    return num / den


results, sources = {}, {}
for name in DATASETS:
    path = load_latest(RESULT_DIRS[name])
    results[name] = json.load(open(path))
    sources[name] = path


print("=== CHECKPOINTS EVALUATED ===")
for dataset in DATASETS:
    print(f"  {dataset:15} {os.path.basename(results[dataset]['model']):<20} "
          f"({os.path.basename(sources[dataset])})")


print("\n=== SECTION SCORES (example-weighted) ===")
print(f"{'Section':<12}" + "".join(f"{d:>16}" for d in DATASETS))
print("-" * 60)
section_scores = {}
for section, subsets in SECTIONS.items():
    section_scores[section] = [
        weighted(results[d]['extra_results'], subsets) for d in DATASETS
    ]
    print(f"{section:<12}" + "".join(f"{v:>16.3f}" for v in section_scores[section]))
print("-" * 60)
print(f"{'OVERALL':<12}" + "".join(f"{results[d]['accuracy']:>16.3f}" for d in DATASETS))


print("\n=== SANITY: below-chance sections (model is anti-correlated, not just weak) ===")
flagged = False
for dataset_idx, dataset in enumerate(DATASETS):
    below = [s for s in SECTIONS if section_scores[s][dataset_idx] < CHANCE]
    if below:
        flagged = True
        print(f"  {dataset:15} below {CHANCE:.2f} on: {', '.join(below)}")
if not flagged:
    print("  none - all sections above chance")


print("\n=== CATEGORY BREAKDOWN ===")
print(f"{'Category':<25}" + "".join(f"{d:>16}" for d in DATASETS))
print("-" * 73)
for cat in sorted(results['hh']['extra_results']):
    print(f"{cat:<25}" + "".join(
        f"{results[d]['extra_results'][cat]:>16.3f}" for d in DATASETS))


print("\n=== SAFETY vs CAPABILITY TRADEOFF ===")
for dataset_idx, dataset in enumerate(DATASETS):
    safety = section_scores['Safety'][dataset_idx]
    reasoning = section_scores['Reasoning'][dataset_idx]
    print(f"  {dataset:15} safety={safety:.3f}  reasoning={reasoning:.3f}")

print("\n  Refusal behavior (does it know when NOT to answer?):")
for dataset in DATASETS:
    extra = results[dataset]['extra_results']
    print(f"    {dataset:15} should-refuse={extra['xstest-should-refuse']:.3f}  "
          f"should-respond={extra['xstest-should-respond']:.3f}  "
          f"dangerous={extra['refusals-dangerous']:.3f}")

best = max(DATASETS, key=lambda x: results[x]['accuracy'])
print(f"\nBest overall: {best} ({results[best]['accuracy']:.4f})")
