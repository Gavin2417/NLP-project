import subprocess

counts = [10, 20, 40]
epochs = [1, 2, 4, 6, 8, 10]

for count in counts:
    for epoch in epochs:
        print(f"\nRunning: python waste_burn_training.py --count {count} --epochs {epoch}")
        result = subprocess.run(
            ["python", "waste_burn_training.py", "--count", str(count), "--epochs", str(epoch)],
            capture_output=True,
            text=True
        )
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)