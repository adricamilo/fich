from matplotlib import pyplot as plt
from pathlib import Path

files = ["training_logs/best/adam dropout lr_scheduler.txt",
         "training_logs/best/adam no dropout 1.txt",
         "training_logs/best/adam no dropout 2.txt",
         "training_logs/best/adam no dropout 3.txt"]

plt.title("Best training sessions 25 first epochs")
plt.ylim(0, 100)
plt.ylabel("Test set accuracy %")
plt.xlabel("Epoch")
plt.axhline(y=100/3, color='y', linestyle='--')

for file in files:
    with open(file) as f:
        raw_lines = f.readlines()
        lines = [float(line.strip()[:-1]) for line in raw_lines[5:]]
        plt.plot(range(1, len(lines)+1), lines, label=Path(file).stem)

plt.legend()
location = "training_logs/best/best.png"
plt.savefig(location)