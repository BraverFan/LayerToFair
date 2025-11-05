import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import numpy as np

RANDOM_STATE = 42

class KeyNeuronsIdentification:
    def __init__(self, model, layer_sizes, X_val, sens_idx, threshold, hyperparameter_search_type, dataset="none", sens_classes=None):
        self.model = model
        model.eval()
        self.layer_sizes = layer_sizes
        self.layers_num = len(layer_sizes)
        self.X_val = X_val
        self.sens_idx = sens_idx
        self.threshold = threshold
        self.hyperparameter_search_type = hyperparameter_search_type
        self.layer_sizes_accuracy = {}
        self.key_neurons = {}
        self.dataset = dataset
        self.sens_classes = sens_classes

        # 处理敏感属性列（将其标准化为0/1标签）
        self._process_sensitive_attribute()

    def _process_sensitive_attribute(self):
        # 保留原始副本
        X_val_processed = self.X_val.clone() if isinstance(self.X_val, torch.Tensor) else self.X_val.copy()
        self.X_val_processed = X_val_processed

        # 处理敏感属性
        if self.dataset == "adult_race":
            # 取出 one-hot 四列（23, 24, 25, 26）表示种族
            sens_val_onehot = self.X_val[:, 23:27]
            # 第26列是 white 的 one-hot 标志
            white_mask = (sens_val_onehot[:, 3] == 1)
            self.sens_val_processed = white_mask.to(dtype=torch.int64)
        elif self.dataset == "meps16":
            sens_val_column = self.X_val[:, self.sens_idx]
            sens_val_column = torch.where(sens_val_column == self.sens_classes[1],
                                          torch.tensor(1),
                                          torch.tensor(0))
            X_val_processed[:, self.sens_idx] = sens_val_column
            print("meps.....")
            self.sens_val_processed = sens_val_column.to(dtype=torch.int64)
        else:
            # 默认：使用 sens_idx 取出某列敏感属性
            sens_val_column = self.X_val[:, self.sens_idx]

            if torch.is_floating_point(sens_val_column) and not torch.all((sens_val_column == 0) | (sens_val_column == 1)):
                mask = torch.isclose(sens_val_column, torch.tensor(1 / 3), atol=1e-6, rtol=1e-6)
                sens_val_column = torch.where(mask, torch.tensor(1), torch.tensor(0))
                X_val_processed[:, self.sens_idx] = sens_val_column
                print("bank.....")

            self.sens_val_processed = sens_val_column.to(dtype=torch.int64)

    def identify(self):
        print("Original X_val:", self.X_val)
        print("Processed X_val:", self.X_val_processed)
        print("Processed sensitive attribute:", self.sens_val_processed)

        # Forward pass to get hidden outputs
        _ = self.model(self.X_val_processed)

        total_train_time = 0
        for i in range(self.layers_num):
            layer_outputs = self.model.hidden_outputs[i]

            if isinstance(layer_outputs, torch.Tensor):
                layer_outputs = layer_outputs.detach().cpu().numpy()

            param_grid = {
                'n_estimators': [i for i in range(50, 201, 25)],
                'max_depth': [i for i in range(3, 11)]# 3, 11
            }

            rf_model = RandomForestClassifier(
                max_features='sqrt',
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                oob_score=True
            )

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            if self.hyperparameter_search_type == 'GridSearch':
                search = GridSearchCV(
                    estimator=rf_model,
                    param_grid=param_grid,
                    cv=skf,
                    scoring='balanced_accuracy',
                    n_jobs=-1
                )
            elif self.hyperparameter_search_type == 'RandomSearch':
                search = RandomizedSearchCV(
                    estimator=rf_model,
                    param_distributions=param_grid,
                    n_iter=20,
                    cv=skf,
                    scoring='balanced_accuracy',
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
            else:
                raise ValueError(f"Unknown hyperparameter search type: {self.hyperparameter_search_type}")

            print("layer_outputs", layer_outputs)
            print("sens_val", self.sens_val_processed.detach().cpu().numpy())

            search.fit(layer_outputs, self.sens_val_processed.detach().cpu().numpy())

            best_index = search.best_index_
            train_time = search.cv_results_['mean_fit_time'][best_index]
            print("Train time:", train_time)

            print(f"\nLayer {i + 1}")
            print("Best Parameters:", search.best_params_)
            print("Best CV Accuracy:", search.best_score_)

            best_model = search.best_estimator_
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            for j in range(self.layer_sizes[i]):
                print(f"Neuron {indices[j]} - Importance: {importances[indices[j]]:.4f}")

            key_neurons = [idx for idx, imp in zip(indices, importances[indices]) if imp > self.threshold]

           #  key_neurons = [idx for idx, imp in zip(indices, importances[indices]) if imp >= 0]

            # threshold_value = np.percentile(importances, 75)
            # key_neurons = [idx for idx, imp in zip(indices, importances[indices]) if imp >= threshold_value]

           # mean_importance = np.mean(importances)
           # print(f"Mean Importance for Layer {i + 1}: {mean_importance:.4f}")
           # key_neurons = [idx for idx, imp in zip(indices, importances[indices]) if imp > mean_importance]
            print("Key Neurons:", key_neurons)

            layer_key = f"layer{i + 1}"
            self.layer_sizes_accuracy[layer_key] = search.best_score_
            self.key_neurons[layer_key] = key_neurons
            total_train_time += train_time

        return self.layer_sizes_accuracy, self.key_neurons, total_train_time

