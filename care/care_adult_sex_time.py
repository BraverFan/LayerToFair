import json
import os
import numpy as np
from models_causal import AdultCensusModel
import torch
from utils import preprocess_adult_census, localize, compute_metrics, convert_to_builtin_types
import pyswarms
import time
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

device = "cpu"


# 适应度函数
def pso_fitness_func(scaling_factors, model, X_val, y_val, sens_val, sens_classes, dataset, baseline_eod, baseline_f1,
                     localize_neurons):
    model.eval()
    result = []
    for i in range(0, int(len(scaling_factors))):
        repair_scaling_factor = scaling_factors[i]
        model.register_scaling_hooks(localize_neurons, repair_scaling_factor)
        eod, dpd, di, f1, acc = compute_metrics(model, X_val, y_val, sens_val, sens_classes, dataset)

        if f1 < 0.98 * baseline_f1:
            fitness = eod + 3.0 * baseline_eod
        else:
            fitness = eod
        # fitness = 0.6 * eod + 0.4 * (1 - f1)
        if f1 >= 0.98 * baseline_f1:
            print(f"Repaired fairness {eod}, f1 {f1}, fitness {fitness}")
        result.append(fitness)

        # 需要清除hooks
        for hook in model.hooks:
            hook.remove()
        model.hooks.clear()
    return result

def main():
    hidden_dims = [64, 128, 256, 256, 256, 256, 128, 64]
    model_paths = [
        'adult-0.6896-982398492-ss-0.2.pt',
        'adult-0.6949-56283202-ss-0.2.pt',
        'adult-0.7004-10989034-ss-0.2.pt',

        'adult-0.7106-72384123-ss-0.2.pt',
        'adult-0.6997-555293809-ss-0.2.pt',
        'adult-0.7005-798457928-ss-0.2.pt',
        'adult-0.6945-66628392-ss-0.2.pt',
        'adult-0.6923-76532891-ss-0.2.pt',

        'adult-0.6918-82398492-ss-0.2.pt',
        'adult-0.6920-232495309-ss-0.2.pt',
    ]

    for model_path in model_paths:
        _, _, seed, _, dropout_rate = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout_rate)
        dataset = "none"
        max_time_minutes = 5
        iters = 5

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_adult_census(seed)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = [0, 1]

        model = AdultCensusModel(p=p).to(device)#注意这里要改
        state_dict = torch.load(f"../saved_models/adult/{model_path}", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

        baseline_eod, baseline_dpd, baseline_di, baseline_f1, baseline_acc = compute_metrics(
            model, X_val, y_val, sens_val, sens_classes, dataset
        )
        print("\nValidation Set Baseline Metrics:")
        print(f"EOD: {baseline_eod:.4f}")
        print(f"DPD: {baseline_dpd:.4f}")
        print(f"DI: {baseline_di:.4f}")
        print(f"F1: {baseline_f1:.4f}")
        print(f"Accuracy: {baseline_acc:.4f}")

        baseline_eod_test, baseline_dpd_test, baseline_di_test, baseline_f1_test, baseline_acc_test = compute_metrics(
            model, X_test, y_test, sens_test, sens_classes, dataset
        )
        print("\nTest Set Baseline Metrics:")
        print(f"EOD: {baseline_eod_test:.4f}")
        print(f"DPD: {baseline_dpd_test:.4f}")
        print(f"DI: {baseline_di_test:.4f}")
        print(f"F1: {baseline_f1_test:.4f}")
        print(f"Accuracy: {baseline_acc_test:.4f}")

        # 因果分析定位神经元（整个神经网络）
        start_time = time.time()
        top_neurons_by_layer = localize(model, X_val, y_val, sens_val, sens_classes, dataset)
        end_time = time.time()
        localize_time = end_time - start_time
        print("top_neurons_by_layer", top_neurons_by_layer)
        print("time for localize neurons with causal:", localize_time)

        # 将定位到的神经元以（“layer1”, 1）这种形式存储,register_scaling_hooks要用到
        localize_neurons = [(layer, neuron) for layer, neurons in top_neurons_by_layer.items() for neuron in neurons]
        repair_neurons_num = sum(len(v) for v in top_neurons_by_layer.values())
        print("repair_neurons_num", repair_neurons_num)
        print("localize_neurons", localize_neurons)

        save_dict = {
            "localize_neurons": localize_neurons,
            "repair_neurons_num": repair_neurons_num,
            "time for localize neurons with causal:": localize_time,
            "baseline_fairness_eod on Val": baseline_eod,
            "baseline_performance on Val": baseline_f1,
            "baseline_accuracy on Val": baseline_acc,
            "baseline_fairness_eod on Test": baseline_eod_test,
            "baseline_performance on Test": baseline_f1_test,
            "baseline_accuracy on Test": baseline_acc_test,
        }

        save_dict = convert_to_builtin_types(save_dict)

        save_dir = f"causal_results/Adult_sex"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{seed}_{max_time_minutes}.json")
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))

        # PSO算法超参数
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}
        # options = {'c1': 1.5, 'c2': 1.5, 'w': 0.8}
        start_time = time.time()
        end_time = start_time + max_time_minutes * 60
        n_particles = 100

        optimizer = pyswarms.single.GlobalBestPSO(n_particles=n_particles, dimensions=repair_neurons_num,
                                                  options=options,
                                                  bounds=([[-1.0] * repair_neurons_num, [1.0] * repair_neurons_num]),
                                                  init_pos=np.zeros((n_particles, repair_neurons_num), dtype=float),
                                                  )

        best_cost = float('inf')
        best_pos = None
        cost_history = []
        total_iters = 0

        while time.time() < end_time:
            current_cost, current_pos = optimizer.optimize(
                pso_fitness_func,
                iters=iters,
                model=model,
                X_val=X_val,
                y_val=y_val,
                sens_val=sens_val,
                sens_classes=sens_classes,
                dataset=dataset,
                baseline_eod=baseline_eod,
                baseline_f1=baseline_f1,
                localize_neurons=localize_neurons
            )

            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = current_pos.copy()

            cost_history.extend(optimizer.cost_history[-10:])
            total_iters += iters

            if time.time() >= end_time - 1:
                break

        print(f"优化完成，总迭代次数: {total_iters}")
        print(f"最佳成本: {best_cost:.4f}")
        print(f"最佳位置: {best_pos}")
        model.register_scaling_hooks(localize_neurons, best_pos)
        best_eod, best_dpd, best_di, best_f1, best_acc = compute_metrics(model, X_val, y_val,
                                                                                                  sens_val,
                                                                                                  sens_classes,
                                                                                                  dataset)
        best_eod_test, best_dpd_test, best_di_test, best_f1_test, best_acc_test = compute_metrics(model, X_test, y_test, sens_test, sens_classes,
                                                                         dataset)
        for hook in model.hooks:
            hook.remove()
        model.hooks.clear()

        save_dict["num_iterations"] = total_iters * n_particles
        save_dict["repaired_fairness_eod_val"] = best_eod
        save_dict["repaired_fairness_f1_val"] = best_f1
        save_dict["repaired_fairness_acc_val"] = best_acc

        save_dict["repaired_fairness_eod_test"] = best_eod_test
        save_dict["repaired_fairness_f1_test"] = best_f1_test
        save_dict["repaired_fairness_acc_test"] = best_acc_test

        save_dict = convert_to_builtin_types(save_dict)

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))

        plot_cost_history(optimizer.cost_history)
        plt.show()


if __name__ == "__main__":
    main()
