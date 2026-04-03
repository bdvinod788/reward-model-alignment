import json

DATASETS = ['hh', 'ultrafeedback', 'mixed']

RESULT_PATHS = {
    'hh':            'outputs/rewardbench/checkpoint-36180.json',
    'ultrafeedback': 'outputs/rewardbench_ultrafeedback/checkpoint-13756.json',
    'mixed':         'outputs/rewardbench_mixed/checkpoint-49936.json',
}

results = {name: json.load(open(path)) for name, path in RESULT_PATHS.items()}


print("=== OVERALL ACCURACY ===")
for dataset in DATASETS:
    print(f"  {dataset:15} {results[dataset]['accuracy']:.4f}")


print("\n=== CATEGORY BREAKDOWN ===")
print(f"{'Category':<30} {'hh':>8} {'ultrafeedback':>15} {'mixed':>8}")
print("-" * 65)
categories = results['hh']['extra_results'].keys()
for cat in categories:
    hh = results['hh']['extra_results'][cat]
    uf = results['ultrafeedback']['extra_results'][cat]
    mx = results['mixed']['extra_results'][cat]
    print(f"{cat:<30} {hh:>8.4f} {uf:>15.4f} {mx:>8.4f}")


print("\n=== TOP 5 BIGGEST DIFFERENCES (HH vs others) ===")
diff = {}
for cat in categories:
    diff[f'hh_vs_uf_{cat}']    = results['hh']['extra_results'][cat] - results['ultrafeedback']['extra_results'][cat]
    diff[f'hh_vs_mixed_{cat}'] = results['hh']['extra_results'][cat] - results['mixed']['extra_results'][cat]

for k, v in sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    direction = "hh > others" if v > 0 else "others > hh"
    print(f"  {k:<40} {v:>+.4f}  ({direction})")


print("\n=== KEY FINDINGS ===")

print("\n1. Safety collapse:")
for cat in ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse']:
    hh = results['hh']['extra_results'][cat]
    uf = results['ultrafeedback']['extra_results'][cat]
    mx = results['mixed']['extra_results'][cat]
    print(f"   {cat:<25} hh={hh:.2f} | uf={uf:.2f} | mixed={mx:.2f}")

print("\n2. Responsiveness tradeoff (xstest-should-respond):")
for dataset in DATASETS:
    print(f"   {dataset:<15} {results[dataset]['extra_results']['xstest-should-respond']:.2f}")

best = max(DATASETS, key=lambda x: results[x]['accuracy'])
print(f"\n3. Best overall dataset: {best} ({results[best]['accuracy']:.4f})")
