import torch
from models import BankModel
from utils import compute_metrics, preprocess_bank
from sa import SimulatedAnnealingRepair


if __name__ == "__main__":
    layer_sizes = [128, 128, 128, 128, 128, 128]
    k_min = 2
    k_max = 230
    sens_classes_all = [
        0.0,
        0.11111111,
        0.22222222,
        0.33333333,
        0.44444444,
        0.55555556,
        0.66666667,
        0.77777778,
        0.88888889,
        1.0,
    ]

    model_paths = [
        # 'bank-0.5471-934827-ss-0.3.pt',
        # 'bank-0.5499-87237482-ss-0.3.pt',
        # 'bank-0.5706-3989056-ss-0.3.pt',
        # 'bank-0.5391-68348022-ss-0.3.pt',
        'bank-0.5633-20182019-ss-0.4.pt',
    ]

    for model_path in model_paths:
        _, _, seed, _, dropout = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout)
        dataset = "none"

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_bank(seed)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = [sens_classes_all[0], sens_classes_all[3]]

        model = BankModel(p=p)
        state_dict = torch.load(f"../saved_models/bank/{model_path}", map_location=torch.device('cpu'))
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

        obj = SimulatedAnnealingRepair(
            model_path=model_path,
            layer_sizes=layer_sizes,
            k_min=k_min,
            k_max=k_max,
            baseline_eod=baseline_eod,
            sens_classes=sens_classes,
            logfile="./sa_runs/Bank/{}_5.log".format(seed),
        )

        obj.estimate_init_temp_and_run(chi_0=0.75, T0=5.0, p=5, eps=1e-3, decay="log")
