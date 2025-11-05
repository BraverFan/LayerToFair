import json
import os
import numpy as np
from models_causal import MEPS16Model
import torch
from utils import preprocess_meps16_tutorial, localize, compute_metrics, convert_to_builtin_types
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
    hidden_dims = [128, 128, 128]
    model_paths = [
        'meps16-0.5237-123-mms-0.2.pt',
        'meps16-0.5073-9230914-mms-0.2.pt',
        'meps16-0.5301-72893492-mms-0.2.pt',
        'meps16-0.5325-2389424-mms-0.2.pt',
        'meps16-0.5396-653131-mms-0.2.pt',

        'meps16-0.5188-3456789-mms-0.2.pt',
        'meps16-0.5133-57289324-mms-0.2.pt',
        'meps16-0.5149-8888234-mms-0.2.pt',
        'meps16-0.5291-2884565-mms-0.2.pt',
        'meps16-0.5159-6573982-mms-0.2.pt',
    ]
    sens_classes_dict = {
        123: [-0.7827223539, 1.2775921822],
        9230914: [-0.7887401581, 1.2678446770],
        72893492: [-0.7804278135, 1.2813484669],
        2389424: [-0.7876764536, 1.2695567608],
        653131: [-0.7987058759, 1.2520253658],

        3456789: [-0.7820159793, 1.2787462473],
        2884565: [-0.7965645790, 1.2553910017],
        57289324: [-0.7851973772, 1.2735651731],
        8888234: [-0.7894497514, 1.2667050362],
        6573982: [-0.7871448398, 1.2704142332]
    }

    for model_path in model_paths:
        _, _, seed, _, dropout_rate = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout_rate)
        dataset = "none"
        max_time_minutes = 5
        iters = 20

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_meps16_tutorial(seed)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = sens_classes_dict[seed]

        model = MEPS16Model(p=p).to(device)
        state_dict = torch.load(f"../saved_models/meps16/{model_path}", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load(f"../saved_models/adult/{model_path}"))
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

        save_dir = f"causal_results/Meps16"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{seed}_{max_time_minutes}.json")
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))

        # PSO算法超参数
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}
        # options = {'c1': 1.5, 'c2': 1.5, 'w': 0.8}
        start_time = time.time()
        end_time = start_time + max_time_minutes * 60
        n_particles = 50

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
        best_eod_test, best_dpd_test, best_di_test, best_f1_test, best_acc_test = compute_metrics(model, X_test, y_test,
                                                                                                  sens_test,
                                                                                                  sens_classes,
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
