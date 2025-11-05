import json
import os
import numpy as np
from scipy import stats

episode = 1000
folder_path = r"C:\Users\fan\Desktop\fsdownload\GRPO\Default-all-layers-again"

total_lengths = []
values = []
cost_times = []

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        fairness_list = data.get("best_fairness_list", [])
        total_length = len(fairness_list)
        total_lengths.append(total_length)

        if total_length >= episode:
            value = fairness_list[episode - 1]
        else:
            value = None  # 如果长度不足，设置为 None
        values.append(value)

        if total_length > 0:
            cost_time = round((300 / (total_length * 10)) * (episode * 10))
        else:
            cost_time = None
        cost_times.append(cost_time)

print("total_lengths:", total_lengths)
print("平均长度是：", np.mean(total_lengths))
print("values:", values)
print("cost_times:", cost_times)

mean = np.mean(values)
std = np.std(values, ddof=1)
n = len(values)
confidence = 0.95
alpha = 1 - confidence
t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
sem = std / np.sqrt(n)
margin = t_crit * sem
ci_lower = mean - margin
ci_upper = mean + margin

mean_pct = mean * 100
margin_pct = margin * 100
ci_lower_pct = ci_lower * 100
ci_upper_pct = ci_upper * 100

print(f"均值: {mean_pct:.3f}%")
print(f"95% 置信区间: {mean_pct:.3f}% ± {margin_pct:.3f}")

print("平均花费时间为:", np.mean(cost_times))

