import torch
from models import CompasModel
from utils import compute_metrics, preprocess_compas
from sa import SimulatedAnnealingRepair

if __name__ == "__main__":
    layer_sizes = [64, 32, 16, 8, 4]
    k_min = 2
    k_max = 35
    sens_classes = [0.0, 1.0]
    model_paths = [
        'compas-0.9684-39823282-mms-0.1.pt',
        'compas-0.9652-923728345-mms-0.1.pt',
        'compas-0.9688-11039402-mms-0.1.pt',
        'compas-0.9712-1945898-mms-0.1.pt',
        'compas-0.9630-13020202-mms-0.1.pt',
        'compas-0.9636-34989052-mms-0.1.pt',
        'compas-0.9659-8883204-mms-0.1.pt',
        'compas-0.9711-3310820954-mms-0.1.pt',
        'compas-0.9697-59923892-mms-0.1.pt',
        'compas-0.9657-66102999-mms-0.1.pt',
    ]

    for model_path in model_paths:
        _, _, seed, _, dropout = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout)
        dataset = "none"
        attr = "race"

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_compas(seed, attr)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = [0, 1]

        model = CompasModel(p=p)
        state_dict = torch.load(f"../saved_models/compas/{model_path}", map_location=torch.device('cpu'))
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

        obj = SimulatedAnnealingRepair(
            model_path=model_path,
            layer_sizes=layer_sizes,
            k_min=k_min,
            k_max=k_max,
            baseline_eod=baseline_eod,
            logfile="./sa_runs/Compas_race/{}_5.log".format(seed),
        )

        obj.estimate_init_temp_and_run(chi_0=0.75, T0=5.0, p=5, eps=1e-3, decay="log")
