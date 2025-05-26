import subprocess
files_name = ['waste_40', 'waste_52', 'waste_77', 'waste_89']
counts = [10, 20, 40]
epochs = [1, 2, 4, 6, 8, 10]
for file_n in files_name:
    for count in counts:
        for epoch in epochs:
            print(f"\nRunning: python waste_burn_training_full.py --count {count} --epochs {epoch} --folder {file_n}")
            result = subprocess.run(
                ["python", "waste_burn_training_half.py", "--count", str(count), "--epochs", str(epoch), "--folder", file_n],
                capture_output=True,
                text=True
            )
            print("Output:\n", result.stdout)
            print("Errors:\n", result.stderr)
            
            print("-----------------------------------------------")

            print(f"\nRunning: python waste_burn_training_half.py --count {count} --epochs {epoch} --folder {file_n}")
            result = subprocess.run(
                ["python", "waste_burn_training_half.py", "--count", str(count), "--epochs", str(epoch), "--folder", file_n],
                capture_output=True,
                text=True
            )
            print("Output:\n", result.stdout)
            print("Errors:\n", result.stderr)

