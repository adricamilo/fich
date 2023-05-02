import os
from matplotlib import pyplot as plt

root_dir = "training_logs"
file_dict = dict()

for dir_, _, files in os.walk(root_dir):
    for file_name in files:
        if file_name.startswith("._") or not file_name.lower().endswith('.txt'):
            continue

        rel_dir = os.path.relpath(dir_, root_dir)
        if rel_dir not in file_dict.keys():
            file_dict[rel_dir] = list()
        file_dict[rel_dir].append(file_name)

for key, value_list in file_dict.items():
    for value in value_list:
        name = os.path.join(root_dir, key, value)
        with open(name) as f:
            raw_lines = f.readlines()
            lines = [float(line.strip()[:-1]) for line in raw_lines[5:]]
            plt.plot(range(1, len(lines)+1), lines)
            plt.title(key)
            plt.ylim(0, 100)
            plt.ylabel("Test set accuracy %")
            plt.xlabel("Epoch")
            plt.text(1, 95, raw_lines[2].strip())
            plt.axhline(y=100/3, color='y', linestyle='--')
            location = os.path.join("log_stats", f"{key}{value[-7:-4]}.png")
            plt.savefig(location)
            plt.close()
