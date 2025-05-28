import subprocess
from tqdm import tqdm

files_name = ['waste_40']
base_counts = [10, 20, 40]
epochs = [1, 2, 4, 7, 10]

# First compute total number of runs (each count/epoch combo runs TWO scripts)

for file_n in files_name:
    get_counts = int(file_n.split('_')[1])
    counts = base_counts.copy()
    if get_counts > counts[-1]:
        counts.append(get_counts)

    for count in counts:
        for epoch in epochs:
            # --- First script (full) ---
            print(f"\nRunning: python waste_burn_training_full.py "
                    f"--count {count} --epochs {epoch} --folder {file_n}")
            result_full = subprocess.run(
                ["python", "waste_burn_training_full.py",
                 "--count", str(count),
                 "--epochs", str(epoch),
                 "--folder", file_n],
                capture_output=True,
                text=True
            )
            print("Output:\n", result_full.stdout)
            print("Errors:\n", result_full.stderr)


            # --- Second script (half) ---
            print(f"\nRunning: python waste_burn_training_half.py "
                  f"--count {count} --epochs {epoch} --folder {file_n}")
            result_half = subprocess.run(
                ["python", "waste_burn_training_half.py",
                 "--count", str(count),
                 "--epochs", str(epoch),
                 "--folder", file_n],
                capture_output=True,
                text=True
            )
            print("Output:\n", result_half.stdout)
            print("Errors:\n", result_half.stderr)


