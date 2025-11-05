import torch
from models import MEPS16Model
from utils import compute_metrics, preprocess_meps16_tutorial
from sa import SimulatedAnnealingRepair


if __name__ == "__main__":
    layer_sizes = [128, 128, 128]
    k_min = 2
    k_max = 115
    model_paths = [
        'meps16-0.5237-123-mms-0.2.pt',
        'meps16-0.5073-9230914-mms-0.2.pt',
        'meps16-0.5182-90234012-mms-0.2.pt',
        'meps16-0.5231-5768121-mms-0.2.pt',
        'meps16-0.5260-3895681-mms-0.2.pt',
        'meps16-0.5261-894221-mms-0.2.pt',
        'meps16-0.5265-4856982-mms-0.2.pt',
        'meps16-0.5301-72893492-mms-0.2.pt',
        'meps16-0.5325-2389424-mms-0.2.pt',
        'meps16-0.5330-4655-mms-0.2.pt',
        'meps16-0.5389-21984392-mms-0.2.pt',
        'meps16-0.5396-653131-mms-0.2.pt',
    ]
    sens_classes_dict = {
        123: [-0.7827223539, 1.2775921822],
        9230914: [-0.7887401581, 1.2678446770],
        90234012: [-0.7883855104, 1.2684149742],
        5768121: [-0.7942484617, 1.2590519190],
        3895681: [-0.7970995903, 1.2545484304],
        894221: [-0.7931807041, 1.2607467175],
        4856982: [-0.7811334729, 1.2801909447],
        72893492: [-0.7804278135, 1.2813484669],
        2389424: [-0.7876764536, 1.2695567608],
        4655: [-0.7880309224, 1.2689857483],
        21984392: [-0.7837826014, 1.2758640051],
        653131: [-0.7987058759, 1.2520253658]
    }

    for model_path in model_paths:
        _, _, seed, _, dropout = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout)
        dataset = "none"

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_meps16_tutorial(seed)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = sens_classes_dict[seed]

        model = MEPS16Model(p=p)
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

        obj = SimulatedAnnealingRepair(
            model_path=model_path,
            layer_sizes=layer_sizes,
            k_min=k_min,
            k_max=k_max,
            baseline_eod=baseline_eod,
            sens_classes=sens_classes,
            logfile="./sa_runs/Meps16/{}_5.log".format(seed),
        )

        obj.estimate_init_temp_and_run(chi_0=0.75, T0=5.0, p=5, eps=1e-3, decay="log")