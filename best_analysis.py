from matplotlib import pyplot as plt
from pathlib import Path

drop = ["training_logs/best/adam dropout lr_scheduler 1.txt",
        "training_logs/best/adam dropout lr_scheduler 2.txt",
        "training_logs/best/adam dropout lr_scheduler 3.txt",
        "training_logs/best/adam dropout lr_scheduler 4.txt"]
no_drop= ["training_logs/best/adam no dropout 1.txt",
          "training_logs/best/adam no dropout 2.txt",
          "training_logs/best/adam no dropout 3.txt"]

plt.title("Best training sessions 25 first epochs")
plt.ylim(0, 100)
plt.ylabel("Test set accuracy %")
plt.xlabel("Epoch")
plt.axhline(y=100/3, color='y', linestyle='--')

i = 1
for file in drop:
    with open(file) as f:
        raw_lines = f.readlines()
        lines = [float(line.strip()[:-1]) for line in raw_lines[5:]]
        plt.plot(range(1, len(lines)+1), lines, label=f"drop {i}")
        i += 1

# i = 1
# for file in no_drop:
#     with open(file) as f:
#         raw_lines = f.readlines()
#         lines = [float(line.strip()[:-1]) for line in raw_lines[5:]]
#         plt.plot(range(1, len(lines)+1), lines, label=f"no drop {i}")
#         i += 1

plt.legend(ncol=3)
location = "training_logs/best/best.png"
plt.show()