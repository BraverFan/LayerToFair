import torch
from models import DefaultModel
from utils import compute_metrics, preprocess_default
from sa import SimulatedAnnealingRepair


if __name__ == "__main__":
    layer_sizes = [64, 32, 16, 8, 4]
    k_min = 2
    k_max = 35
    model_paths = [
        'default-0.4915-51291923-mms-0.2.pt',
        'default-0.4936-68394093-mms-0.2.pt',
         'default-0.4947-9023472-mms-0.2.pt',
         'default-0.4957-723984892-mms-0.2.pt',
         'default-0.4958-442391456-mms-0.2.pt',
         'default-0.5048-889283267-mms-0.2.pt',
         'default-0.5060-67729323-mms-0.2.pt',
         'default-0.5068-148928732-mms-0.2.pt',
         'default-0.5102-34820934-mms-0.2.pt',
         'default-0.5133-99916872-mms-0.2.pt',
    ]

    for model_path in model_paths:
        _, _, seed, _, dropout = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout)
        dataset = "none"

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_default(seed)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = [0.0, 1.0]

        model = DefaultModel(p=p)
        state_dict = torch.load(f"../saved_models/default/{model_path}", map_location=torch.device('cpu'))
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
            logfile="./sa_runs/Default/{}_5.log".format(seed),
        )

        obj.estimate_init_temp_and_run(chi_0=0.75, T0=5.0, p=5, eps=1e-3, decay="log")