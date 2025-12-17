import json

with open('results/mnist_hpo_comparison_results.json', 'r') as f:
    data = json.load(f)

print(f"\nTotal experiments: {len(data)}/12")
print("\nCompleted experiments:")
for i, r in enumerate(data, 1):
    print(f"  {i}. {r['model']:15s} - {r['hpo_method']:25s} (acc: {r['test_accuracy']:.4f})")

models = set(r['model'] for r in data)
print(f"\nModels: {models}")

# CNN 확인
cnn_results = [r for r in data if r['model'] == 'cnn']
print(f"\nCNN experiments: {len(cnn_results)}/4")
if cnn_results:
    for r in cnn_results:
        print(f"  - {r['hpo_method']}: {r['test_accuracy']:.4f}")
else:
    print("  (No CNN results found)")
