import copy
import os.path
import pandas as pd
import torch
import numpy as np
from aif360.datasets import MEPSDataset21
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
)
from typing import Tuple
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adult_race_to_ohe = {
    "Amer-Indian-Eskimo": [0.0, 0.0, 0.0, 0.0],
    "Asian-Pac-Islander": [1.0, 0.0, 0.0, 0.0],
    "Black": [0.0, 1.0, 0.0, 0.0],
    "Other": [0.0, 0.0, 1.0, 0.0],
    "White": [0.0, 0.0, 0.0, 1.0],
}


def convert_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# def localize(model, X_val, y_val, sens_val, sens_classes, dataset, baseline_eod):
#     model.eval()
#     neuron_max_min = {}
#     neuron_responsibility = {}
#     total_neurons = 0
#     _ = model(X_val)
#     for i, activations in enumerate(model.hidden_outputs):
#         layer_name = f"layer{i + 1}"
#         if layer_name not in neuron_max_min:
#             neuron_max_min[layer_name] = []
#         num_neurons = activations.shape[1]
#         total_neurons += num_neurons
#         for neuron_idx in range(num_neurons):
#             neuron_activations = activations[:, neuron_idx]
#             min_val = torch.min(neuron_activations).item()
#             max_val = torch.max(neuron_activations).item()
#             neuron_max_min[layer_name].append((neuron_idx, min_val, max_val))
#
#     for layer_name in neuron_max_min:
#         model = model.eval()
#         if layer_name not in neuron_responsibility:
#             neuron_responsibility[layer_name] = []
#
#         for neuron_idx in range(len(neuron_max_min[layer_name])):
#             neuron_min = neuron_max_min[layer_name][neuron_idx][1]
#             neuron_max = neuron_max_min[layer_name][neuron_idx][2]
#             intervention_values = torch.linspace(neuron_min, neuron_max, 20)
#
#             fairness_interventions = []
#             for do_value in intervention_values:
#                 model.register_do_hooks(layer_name, neuron_idx, do_value)
#                 with torch.no_grad():
#                     y_pred = (torch.sigmoid(model(X_val)).view(-1) > 0.5).float()
#                     new_fairness = 1 - equal_opp_difference(y_val, y_pred, sens_val, sens_classes, dataset)
#                     fairness_interventions.append(new_fairness)
#             neuron_responsibility[layer_name].append((neuron_idx, np.mean(fairness_interventions)))
#     for hook in model.hooks:
#         hook.remove()
#     model.hooks.clear()
#
#     # 取出效应值大于baseline的神经元
#     count = 0
#     layer_avg_above_baseline_neurons = {}
#     for layer_name in neuron_responsibility:
#         if layer_name not in layer_avg_above_baseline_neurons:
#             layer_avg_above_baseline_neurons[layer_name] = []
#         for item in neuron_responsibility[layer_name]:
#             if item[1] > baseline_eod:
#                 count += 1
#                 layer_avg_above_baseline_neurons[layer_name].append((item[0], item[1]))
#     print(f"效应值大于baseline的神经元有{count} / {total_neurons}个")
#
#     neuron_responsibility = layer_avg_above_baseline_neurons
#
#     # 取出效应值top-k个神经元
#     all_neurons = []
#     for layer_name in neuron_responsibility:
#         for neuron_info in neuron_responsibility[layer_name]:
#             neuron_idx, responsibility = neuron_info
#             all_neurons.append((layer_name, neuron_idx, responsibility))
#     all_neurons.sort(key=lambda x: x[2], reverse=True)
#     print("sorted_all_neurons", all_neurons)
#     top_n = int(total_neurons * 0.1)
#     top_neurons = all_neurons[:top_n]
#     top_neurons_by_layer = {}
#     for layer_name, neuron_idx, responsibility in top_neurons:
#         if layer_name not in top_neurons_by_layer:
#             top_neurons_by_layer[layer_name] = []
#         top_neurons_by_layer[layer_name].append(neuron_idx)
#
#     # 每一层中选择效应值最大的若干个神经元
#     # top_neurons_by_layer = {}
#     # for layer_name in neuron_responsibility:
#     #     if layer_name == "layer1":
#     #         top_n_layer = 23
#     #     if layer_name == "layer2":
#     #         top_n_layer = 19
#     #     # if layer_name == "layer3":
#     #     #     top_n_layer = 20
#     #     # if layer_name == "laye4":
#     #     #     top_n_layer = 23
#     #     layer_candidates = neuron_responsibility[layer_name]
#     #     sorted_layer = sorted(layer_candidates, key=lambda x: x[1], reverse=True)
#     #     selected_neurons = sorted_layer[:top_n_layer]
#     #     top_neurons_by_layer[layer_name] = [neuron[0] for neuron in selected_neurons]
#
#     return top_neurons_by_layer


def localize(model, X_val, y_val, sens_val, sens_classes, dataset, neurons_num=None, key_layers=None):
    model.eval()
    neuron_max_min = {}
    neuron_responsibility = {}
    total_neurons = 0
    _ = model(X_val)
    for i, outputs in enumerate(model.hidden_outputs):
        layer_name = f"layer{i + 1}"
        if layer_name not in neuron_max_min:
            neuron_max_min[layer_name] = []
        num_neurons = outputs.shape[1]
        total_neurons += num_neurons
        for neuron_idx in range(num_neurons):
            neuron_outputs = outputs[:, neuron_idx]
            min_val = torch.min(neuron_outputs).item()
            max_val = torch.max(neuron_outputs).item()
            neuron_max_min[layer_name].append((neuron_idx, min_val, max_val))

    for layer_name in neuron_max_min:
        model = model.eval()
        if layer_name not in neuron_responsibility:
            neuron_responsibility[layer_name] = []

        results = Parallel(n_jobs=8)(  # 使用全部CPU核
            delayed(evaluate_neuron_effect_single)(
                model, X_val, y_val, sens_val, sens_classes, dataset,
                layer_name, neuron_idx, neuron_min, neuron_max
            )
            for neuron_idx, neuron_min, neuron_max in neuron_max_min[layer_name]
        )

        neuron_responsibility[layer_name] = results
    for hook in model.hooks:
        hook.remove()
    model.hooks.clear()

    if key_layers is None:
        key_layers = len(neuron_responsibility)

   # all_neurons = []
   # for layer_name in neuron_responsibility:
   #     for neuron_info in neuron_responsibility[layer_name]:
   #         neuron_idx, responsibility = neuron_info
   #         all_neurons.append((layer_name, neuron_idx, responsibility))

   # all_neurons.sort(key=lambda x: x[2], reverse=True)
   # print("sorted_all_neurons", all_neurons)

   # top_n = int(total_neurons * 0.1)
   # top_neurons = all_neurons[:top_n]

   # top_neurons_by_layer = {}
   # for layer_name, neuron_idx, responsibility in top_neurons:
   #     if layer_name not in top_neurons_by_layer:
   #         top_neurons_by_layer[layer_name] = []
   #     top_neurons_by_layer[layer_name].append(neuron_idx)
    
    all_neurons = []
    for i in range(key_layers):
        layer_name = f"layer{i + 1}"
        if layer_name not in neuron_responsibility:
            continue
        for neuron_idx, responsibility in neuron_responsibility[layer_name]:
            all_neurons.append((layer_name, neuron_idx, responsibility))
    print(all_neurons)
    all_neurons.sort(key=lambda x: x[2], reverse=True)
    if neurons_num is None:
        print("-----")
        top_n = int(len(all_neurons) * 0.1)
    else:
        print("*****")
        top_n = neurons_num
    top_neurons = all_neurons[:top_n]

    top_neurons_by_layer = {}
    for layer_name, neuron_idx, _ in top_neurons:
        if layer_name not in top_neurons_by_layer:
            top_neurons_by_layer[layer_name] = []
        top_neurons_by_layer[layer_name].append(neuron_idx)


    return top_neurons_by_layer


def evaluate_neuron_effect_single(model, X_val, y_val, sens_val, sens_classes, dataset,
                                  layer_name, neuron_idx, neuron_min, neuron_max):
    local_model = copy.deepcopy(model)
    local_model.eval()

    intervention_values = torch.linspace(neuron_min, neuron_max, 16)  # 粒度越小越快
    fairness_interventions = []

    for do_value in intervention_values:
        local_model.register_do_hooks(layer_name, neuron_idx, do_value)
        with torch.no_grad():
           # y_pred = (torch.sigmoid(local_model(X_val)).view(-1) > 0.5).float()
           # new_fairness = 1 - equal_opp_difference(y_val, y_pred, sens_val, sens_classes, dataset)
            new_fairness, _, _, _, _ = compute_metrics(local_model, X_val, y_val, sens_val, sens_classes, dataset)
            fairness_interventions.append(new_fairness)

        for hook in local_model.hooks:
            hook.remove()
        local_model.hooks.clear()

    return (neuron_idx, np.mean(fairness_interventions))

def compute_metrics(model, X_val, y_val, sens_val, sens_classes, dataset):
    model = model.eval()
    y_pred = (torch.sigmoid(model(X_val)).view(-1) > 0.5).float().cpu().numpy()
    y_val = y_val.cpu().numpy()
    sens_val = sens_val.cpu().numpy()
    baseline_eod = 1 - equal_opp_difference(y_val, y_pred, sens_val, sens_classes, dataset)
    baseline_dpd = 1 - demographic_parity_difference(y_pred, sens_val, sens_classes, dataset)
    baseline_di = disparate_impact(y_pred, sens_val, sens_classes, dataset)
    baseline_f1 = f1_score(y_val, y_pred)
    baseline_acc = accuracy_score(y_val, y_pred)

    return baseline_eod, baseline_dpd, baseline_di, baseline_f1, baseline_acc


def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + 'is already exists!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + 'create successfully!')


def load_adult_census(dataset: str = "adult") -> Tuple[np.ndarray, np.ndarray]:
    if dataset == "adult":
        df = pd.read_csv("data/adult.csv")

    # Replace unknown values with NaN
    replace_chars = ["\n", "\n?\n", "?", "\n?", " ?", "? ", " ? ", " ?\n"]
    if any(char in df.values for char in replace_chars):
        df.replace(replace_chars, np.nan, inplace=True)

    df = df.fillna("Missing")

    df.columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    df["occupation"] = df["occupation"].str.strip()

    df["new_occupation"] = df["occupation"].replace(
        {
            "Prof-specialty": "Professional_Managerial",
            "Craft-repair": "Skilled_Technical",
            "Exec-managerial": "Professional_Managerial",
            "Adm-clerical": "Sales_Administrative",
            "Sales": "Sales_Administrative",
            "Other-service": "Service_Care",
            "Machine-op-inspct": "Skilled_Technical",
            "Missing": "Unclassified Occupations",
            "Transport-moving": "Skilled_Technical",
            "Handlers-cleaners": "Service_Care",
            "Farming-fishing": "Service_Care",
            "Tech-support": "Skilled_Technical",
            "Protective-serv": "Professional_Managerial",
            "Priv-house-serv": "Service_Care",
            "Armed-Forces": "Unclassified Occupations",
        }
    )

    df.drop(["occupation"], axis=1, inplace=True)  # 删除这个列
    df.rename(columns={"new_occupation": "occupation"}, inplace=True)

    data_types = {
        "age": "uint8",
        "workclass": "category",
        "fnlwgt": "int32",
        "education": "category",
        "education-num": "uint8",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital-gain": "int32",
        "capital-loss": "int32",
        "hours-per-week": "uint8",
        "native-country": "category",
        "income": "category",
    }
    df = df.astype(data_types)  # 转换数据类型

    # drop education and native country columns
    df.drop(["education"], axis=1, inplace=True)
    df.drop(["native-country"], axis=1, inplace=True)

    race_label_encoder = LabelEncoder()  # 这个是把分类特征转换为对应的一个整数0、1、2

    df["workclass"] = LabelEncoder().fit_transform(df["workclass"])
    df["marital-status"] = LabelEncoder().fit_transform(df["marital-status"])
    df["occupation"] = LabelEncoder().fit_transform(df["occupation"])
    df["relationship"] = LabelEncoder().fit_transform(df["relationship"])
    race_label_encoder.fit(df["race"])
    df["race"] = race_label_encoder.transform(df["race"])
    df["sex"] = LabelEncoder().fit_transform(df["sex"])

    ohe1 = OneHotEncoder(drop="first")  # drop="first"代表只有一个哑变量，如果没有指明drop，则默认为None
    ohe2 = OneHotEncoder(drop="first")
    ohe3 = OneHotEncoder(drop="first")
    ohe4 = OneHotEncoder(drop="first")
    ohe5 = OneHotEncoder(drop="first")
    ohe6 = OneHotEncoder(drop="first")

    # Fit and transform the categorical features using one-hot encoding
    workclass_encoded = ohe1.fit_transform(df[["workclass"]]).toarray()  # 注意这里要转换为密集数组
    marital_encoded = ohe2.fit_transform(df[["marital-status"]]).toarray()
    occupation_encoded = ohe3.fit_transform(df[["occupation"]]).toarray()
    relationship_encoded = ohe4.fit_transform(df[["relationship"]]).toarray()
    race_encoded = ohe5.fit_transform(df[["race"]]).toarray()
    sex_encoded = ohe6.fit_transform(df[["sex"]]).toarray()

    # Convert the encoded features to pandas DataFrames
    workclass_array = pd.DataFrame(
        workclass_encoded, columns=ohe1.get_feature_names_out()
    )
    marital_array = pd.DataFrame(marital_encoded, columns=ohe2.get_feature_names_out())
    occupation_array = pd.DataFrame(
        occupation_encoded, columns=ohe3.get_feature_names_out()
    )
    relationship_array = pd.DataFrame(
        relationship_encoded, columns=ohe4.get_feature_names_out()
    )
    race_array = pd.DataFrame(race_encoded, columns=ohe5.get_feature_names_out())
    sex_array = pd.DataFrame(sex_encoded, columns=ohe6.get_feature_names_out())

    # Drop the original categorical features
    df_dropped = df.drop(
        ["workclass", "marital-status", "occupation", "relationship", "race", "sex"],
        axis=1,
    )

    # Concatenate the encoded features with the numerical features
    df_encoded = pd.concat(
        [
            workclass_array,
            marital_array,
            occupation_array,
            relationship_array,
            race_array,
            sex_array,
            df_dropped,
        ],
        axis=1,
    )

    df_encoded["income"] = LabelEncoder().fit_transform(df_encoded["income"])

    X = df_encoded.drop(["income"], axis=1).to_numpy()
    y = df_encoded["income"].to_numpy()
    # # 新添加的，把X和y分别保存成csv文件方便查看
    # X_df = pd.DataFrame(X)
    # y_df = pd.DataFrame(y, columns=["income"])
    # df = pd.concat([X_df, y_df], axis=1)
    # df.to_csv("data/df_enconded_adult.csv", index=False)  # index=False 参数用于不保存 DataFrame 的索引列
    return X, y


def equal_opp_difference(
        y_true, y_pred, sensi_feat, sens_classes=[0, 1], dataset="none"
):
    # print("y_true",y_true)
    # print("y_pred",y_pred)
    # print("sensi_feat",sensi_feat)
    # print("sens_classes",sens_classes)
    # print("dataset",dataset)
    if dataset != "none":
        return equal_opp_difference_multi(y_true, y_pred, sensi_feat, dataset)

    error_rates = {}

    for i in sens_classes:  # 0、1   [0.0, 0.33333]
        error_rates[i] = {}
        for j in [0, 1]:
            idx = (sensi_feat == i) & (y_true == j)
            # idx = (np.isclose(sensi_feat, i)) & (y_true == j)#只有meps数据集需要这样

            expc = y_pred[idx].mean().item()
            if np.isnan(expc):
                expc = 0.0
            error_rates[i][j] = expc

    # Change this
    tprs = []
    fprs = []
    for cls in sens_classes:
        tprs.append(error_rates[cls][1])
        fprs.append(error_rates[cls][0])

    tpr_diff = max(tprs) - min(tprs)  # 标签值都为1，不同敏感属性预测为1的样本占比的差异
    fpr_diff = max(fprs) - min(fprs)  # 标签值都为0，不同敏感属性预测为1的样本占比的差异

    return 1 - max(tpr_diff, fpr_diff)


def demographic_parity_difference(y_pred, sensi_feat, sens_classes=[0, 1], dataset="none"):
    """
    计算 Demographic Parity (DP) 差异：
    DP = P(ŷ=1 | S=0) - P(ŷ=1 | S=1)

    参数：
    - y_pred: 模型预测值，张量或numpy数组，取值为0或1
    - sensi_feat: 敏感属性，张量或numpy数组
    - sens_classes: 敏感属性的两个类别，默认[0, 1]

    返回：
    - DP 差异的绝对值（越小越公平）
    """

    if dataset != "none":
        return demographic_parity_difference_multi(y_pred, sensi_feat, dataset)

    positive_rates = {}

    for cls in sens_classes:
        idx = sensi_feat == cls
        rate = y_pred[idx].mean().item() if np.sum(idx) > 0 else 0.0
        if np.isnan(rate):
            rate = 0.0
        positive_rates[cls] = rate

    dp_diff = abs(positive_rates[sens_classes[0]] - positive_rates[sens_classes[1]])
    return 1 - dp_diff


def disparate_impact(y_pred, sensi_feat, sens_classes=[0, 1], dataset="none"):
    """
    计算 Disparate Impact (DI)
    DI = min(
        P(ŷ = 1 | S = 0) / P(ŷ = 1 | S = 1),
        P(ŷ = 1 | S = 1) / P(ŷ = 1 | S = 0)
    )

    参数:
        y_pred: 模型的预测值 (0 或 1)，形状为 (n,)
        sensi_feat: 敏感属性特征，形状为 (n,)
        sens_classes: 敏感属性的两个类别，默认是 [0, 1]

    返回:
        DI: Disparate Impact 值，越接近1越公平
    """

    if dataset != "none":
        return disparate_impact_multi(y_pred, sensi_feat, dataset)

    # 获取两个敏感群体的索引
    idx_0 = sensi_feat == sens_classes[0]
    idx_1 = sensi_feat == sens_classes[1]

    # 计算每个群体中被预测为正类的比例
    p0 = y_pred[idx_0].mean().item() if idx_0.sum() > 0 else 0.0
    p1 = y_pred[idx_1].mean().item() if idx_1.sum() > 0 else 0.0

    # 避免除0错误
    if p0 == 0 or p1 == 0:
        return 0.0  # 极不公平

    # 计算 Disparate Impact（取较小的那个比值）
    di = min(p0 / p1, p1 / p0)
    return di


def equal_opp_difference_multi(y_true, y_pred, sensi_feat, dataset="adult_sex"):
    error_rates = {}

    sens_classes = [False, True]

    if dataset == "adult_race":
        sens_attr = np.array(adult_race_to_ohe["White"])

    for i in sens_classes:
        error_rates[i] = {}
        for j in [0, 1]:
            mask_np = (np.prod(sensi_feat == sens_attr, axis=-1) == i)
            mask_torch = torch.from_numpy(mask_np).bool()
            y_true_bool = (y_true == j).astype(bool)
            y_true_torch = torch.from_numpy(y_true_bool).bool()
            idx = mask_torch & y_true_torch
            expc = y_pred[idx].mean().item()
            if np.isnan(expc):
                expc = 0.0
            error_rates[i][j] = expc


# Change this
    tprs = []
    fprs = []
    for cls in sens_classes:
        tprs.append(error_rates[cls][1])
        fprs.append(error_rates[cls][0])

    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)

    return 1 - max(tpr_diff, fpr_diff)

def demographic_parity_difference_multi(y_pred, sensi_feat, dataset="adult_race"):
    error_rates = {}
    sens_classes = [False, True]

    if dataset == "adult_race":
        sens_attr = np.array(adult_race_to_ohe["White"])  # 以 White 为参照组

    for cls in sens_classes:
        mask_np = (np.prod(sensi_feat == sens_attr, axis=-1) == cls)
        mask_torch = torch.from_numpy(mask_np).bool()
        if mask_torch.sum() == 0:
            rate = 0.0
        else:
            rate = y_pred[mask_torch].mean().item()
        if np.isnan(rate):
            rate = 0.0
        error_rates[cls] = rate

    dp_diff = abs(error_rates[False] - error_rates[True])
    return 1 - dp_diff

def disparate_impact_multi(y_pred, sensi_feat, dataset="adult_race"):
    sens_classes = [False, True]

    if dataset == "adult_race":
        sens_attr = np.array(adult_race_to_ohe["White"])  # 以 White 为参照族群

    rates = {}
    for cls in sens_classes:
        mask_np = (np.prod(sensi_feat == sens_attr, axis=-1) == cls)
        mask_torch = torch.from_numpy(mask_np).bool()
        if mask_torch.sum() == 0:
            rate = 0.0
        else:
            rate = y_pred[mask_torch].mean().item()
        if np.isnan(rate):
            rate = 0.0
        rates[cls] = rate

    p0, p1 = rates[False], rates[True]

    if p0 == 0 or p1 == 0:
        return 0.0

    return min(p0 / p1, p1 / p0)


def preprocess_adult_census(
        seed: int = 123, scaler: str = "ss"
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    X, y = load_adult_census(dataset="adult")  # 这里的X是独热编码之后的所有特征，y是标签对应0和1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )  # 分样本的时候要用到随机种子确保和训练的时候分到的样本一样对于验证集
    # 训练、验证、test是6 2 2
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.3, random_state=seed
    # )

    columns_to_standardize = [28, 29, 30, 31, 32, 33]

    # Create a StandardScaler object
    scaler = StandardScaler()

    scaler.fit(X_train[:, columns_to_standardize])

    X_train[:, columns_to_standardize] = scaler.transform(
        X_train[:, columns_to_standardize]
    )
    X_val[:, columns_to_standardize] = scaler.transform(
        X_val[:, columns_to_standardize]
    )
    X_test[:, columns_to_standardize] = scaler.transform(
        X_test[:, columns_to_standardize]
    )

    # Convert the NumPy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_val = torch.from_numpy(X_val).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)

    return X_train, X_val, X_test, y_train, y_val, y_test, 27


def preprocess_bank(seed: int = 42, scaler: str = "mms"):
    df = pd.read_csv("./data/bank-full.csv", sep=";")

    cat_columns_oh = [
        "job",
        "marital",
        "education",
        "contact",
        "poutcome",
    ]

    cat_columns_mms = [
        "age",
        "day",
        "month",
    ]

    num_columns = [
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]

    binary = [
        "default",
        "housing",
        "loan",
    ]

    # Age will be bucketed into 10 buckets
    # 将 df 数据框中的 age 列分成 10 个区间，并用从 0 到 9 的标签对这些区间进行标记
    df["age"] = pd.cut(df["age"], 10, labels=[i for i in range(0, 10)])

    # Convert month from string to int
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    df["month"] = df["month"].map(months)

    # One Hot encode all the cateogrical columns
    oh_encoder = OneHotEncoder(drop="first")
    encoded_data = oh_encoder.fit_transform(df[cat_columns_oh]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=oh_encoder.get_feature_names_out())
    df = df.join(encoded_df)
    df = df.drop(columns=cat_columns_oh)

    # Convert binary columns to 0/1
    df[binary] = df[binary].replace({"no": 0, "yes": 1})

    # convert y to 0/1
    df["y"] = df["y"].replace({"no": 0, "yes": 1})

    # MinMax scale all the mms columns
    mms_scaler = MinMaxScaler(feature_range=(0, 1))
    df[cat_columns_mms] = mms_scaler.fit_transform(df[cat_columns_mms])
    # drop不会修改原来的数据框而是返回一个新的数据框
    X = df.drop(columns=["y"])
    y = df["y"]

    # Get the index for the sensitive feature (age)
    # 获取某列的索引
    sens_idx = X.columns.get_loc("age")  # 这就是为什么sens_Classes是0-0.33333333

    # Create data splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.3, random_state=seed
    # )

    # Use a standard scaler for the remaining columns
    ss = StandardScaler()
    X_train[num_columns] = ss.fit_transform(X_train[num_columns])
    X_test[num_columns] = ss.transform(X_test[num_columns])
    X_val[num_columns] = ss.transform(X_val[num_columns])

    # Convert numpy arrays to tensors
    X_train = torch.from_numpy(X_train.to_numpy()).type(torch.float)
    X_test = torch.from_numpy(X_test.to_numpy()).type(torch.float)
    X_val = torch.from_numpy(X_val.to_numpy()).type(torch.float)
    y_train = torch.from_numpy(y_train.to_numpy()).type(torch.float)
    y_test = torch.from_numpy(y_test.to_numpy()).type(torch.float)
    y_val = torch.from_numpy(y_val.to_numpy()).type(torch.float)

    return X_train, X_val, X_test, y_train, y_val, y_test, sens_idx


def preprocess_compas(
        seed: int = 42, attr="race", scaler: str = "mms"
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """
    Prepare the data of dataset Compas
    :return: X, Y, input shape and number of classes
    sensitive_param == 3
    """
    X = []
    Y = []
    i = 0
    with open("./data/compas", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(",")
            if i == 0:
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                # Y.append([1, 0])
                Y.append(0)
            else:
                # Y.append([0, 1])
                Y.append(1)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    if scaler == "mms":
        mms = MinMaxScaler(feature_range=(0, 1))
        # It is expected to know the min-max so this is fine
        X = mms.fit_transform(X)
    else:
        print("DEPRECATED")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, Y, test_size=0.3, random_state=seed
    # )

    # Convert the NumPy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_val = torch.from_numpy(X_val).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)

    if attr == "race":
        sens_idx = 2
    elif attr == "sex":
        sens_idx = 0
    else:
        raise ValueError("Invalid attr")

    return X_train, X_val, X_test, y_train, y_val, y_test, sens_idx


def preprocess_default(seed: int = 42, scaler: str = "mms"):
    df = pd.read_csv("./data/UCI_Credit_Card.csv")
    df = df.rename(columns={"PAY_0": "PAY_1"})
    df = df.drop(columns=["ID"])
    df = df.astype(float)
    all_dependent_columns = df.columns.tolist()
    all_dependent_columns.remove("default.payment.next.month")

    cat_columns_oh = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]  # Categorical columns that need to be one hot
    cat_columns_mms = [
        "PAY_1",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    num_columns = [
        column
        for column in all_dependent_columns
        if column not in cat_columns_oh + cat_columns_mms
    ]  # Numerical columns

    # One Hot encode all the cateogrical columns
    oh_encoder = OneHotEncoder(drop="first")
    encoded_data = oh_encoder.fit_transform(df[cat_columns_oh]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=oh_encoder.get_feature_names_out())
    df = df.join(encoded_df)
    df = df.drop(columns=cat_columns_oh)

    # MinMax scale all the mms columns
    mms_scaler = MinMaxScaler(feature_range=(0, 1))
    df[cat_columns_mms] = mms_scaler.fit_transform(df[cat_columns_mms])

    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]

    # Get the index for the sensitive feature (sex)
    sens_idx = X.columns.get_loc("SEX_2.0")

    # Create data splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.3, random_state=seed
    # )

    # Use a standard scaler for the remaining columns
    ss = StandardScaler()
    X_train[num_columns] = ss.fit_transform(X_train[num_columns])
    X_test[num_columns] = ss.transform(X_test[num_columns])
    X_val[num_columns] = ss.transform(X_val[num_columns])

    # Convert numpy arrays to tensors
    X_train = torch.from_numpy(X_train.to_numpy()).type(torch.float)
    X_test = torch.from_numpy(X_test.to_numpy()).type(torch.float)
    X_val = torch.from_numpy(X_val.to_numpy()).type(torch.float)
    y_train = torch.from_numpy(y_train.to_numpy()).type(torch.float)
    y_test = torch.from_numpy(y_test.to_numpy()).type(torch.float)
    y_val = torch.from_numpy(y_val.to_numpy()).type(torch.float)

    return X_train, X_val, X_test, y_train, y_val, y_test, sens_idx


def preprocess_meps16_tutorial(seed: int = 42, scaler: str = "mms"):
    # The MEPS16 tutorial on AIF360 uses the StandardScaler on all features
    # We follow that approach here
    cd = MEPSDataset21()
    df = pd.DataFrame(cd.features)

    X = np.array(df.to_numpy(), dtype=float)
    Y = np.array(cd.labels, dtype=int)
    # Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=int).squeeze()
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, Y, test_size=0.3, random_state=seed
    # )

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    X_val = ss.transform(X_val)

    # Convert the NumPy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_val = torch.from_numpy(X_val).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)

    torch.set_printoptions(precision=10)
    print(X_train[:10])

    return X_train, X_val, X_test, y_train, y_val, y_test, 1

def compute_neuron_forward_impact(model, X_val, batch_size=128, key_layers_num=None):
    model.eval()
    grad_sums = None
    total_samples = 0
    dataloader = DataLoader(X_val, batch_size=batch_size, shuffle=False)

    if key_layers_num is None:
        num_layers = len(model.hidden_outputs)
    else:
        num_layers = key_layers_num

    for x_batch in dataloader:
        model.zero_grad()
        output = model(x_batch).squeeze()
        grads_per_layer = []

        for layer_idx in range(num_layers):
            hidden = model.hidden_outputs[layer_idx]
            grad = torch.autograd.grad(
                outputs=output.sum(),
                inputs=hidden,
                retain_graph=(layer_idx != num_layers - 1),
                allow_unused=True
            )[0]
            if grad is None:
                grad = torch.zeros_like(hidden)

            grad_abs = grad.detach().abs()
            grad_norm = grad_abs / (grad_abs.sum(dim=1, keepdim=True) + 1e-8)

            grads_per_layer.append(grad_norm)

        if grad_sums is None:
            grad_sums = [torch.zeros_like(g[0]) for g in grads_per_layer]

        for i, grad_norm in enumerate(grads_per_layer):
            grad_sums[i] += grad_norm.sum(dim=0).cpu()

        total_samples += x_batch.size(0)

    avg_impacts = [layer_sum / total_samples for layer_sum in grad_sums]

    impacts_dict = {}
    for i, layer_impact in enumerate(avg_impacts):
        layer_name = f"layer{i + 1}"
        impacts_dict[layer_name] = layer_impact.tolist()

    impact_tuples = []
    for layer_name, impacts in impacts_dict.items():
        for neuron_idx, impact_score in enumerate(impacts):
            impact_tuples.append((layer_name, neuron_idx, impact_score))

    impact_tuples_sorted = sorted(impact_tuples, key=lambda x: x[2], reverse=True)

    return impact_tuples_sorted
