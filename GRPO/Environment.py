import numpy as np
from utils import compute_metrics

class NeuronScalingEnv:
    def __init__(
            self,
            model,
            candidate_neurons,
            baseline_fairness,
            baseline_performance,
            X_val,
            y_val,
            sens_val,
            sens_classes,
            dataset,
            performance_threshold=0.98,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.candidate_neurons = candidate_neurons
        self.baseline_fairness = baseline_fairness
        self.baseline_performance = baseline_performance
        self.performance_threshold = performance_threshold
        self.sens_classes = sens_classes
        self.dataset = dataset
        self.X_val = X_val
        self.y_val = y_val
        self.sens_val = sens_val

        self.best_fairness = baseline_fairness
        self.best_performance = baseline_performance
        self.best_scales = np.zeros(len(self.candidate_neurons))

        self.current_scales = np.zeros(len(self.candidate_neurons))
        self.current_fairness = baseline_fairness
        self.current_performance = baseline_performance

    def apply_scaling(self, scaling_factors):
         self.model.register_scaling_hooks(self.candidate_neurons, scaling_factors)

    def reset(self):
        self.current_scales = np.zeros(len(self.candidate_neurons))
        self.current_fairness = self.baseline_fairness
        self.current_performance = self.baseline_performance
        self.apply_scaling(self.current_scales)

        return np.concatenate([
            self.current_scales,
            [self.current_fairness, self.current_performance]
        ])

    def step(self, action):
        self.apply_scaling(action)
        self.current_scales = action

        new_fairness, _, _, new_performance, _ = compute_metrics(
            self.model, self.X_val, self.y_val,
            self.sens_val, self.sens_classes, self.dataset
        )
        print("new_fairness", new_fairness)
        print("new_performance", new_performance)

        reward = 0.5 * (1-new_fairness) + 0.5 * new_performance

        if new_performance >= self.performance_threshold * self.baseline_performance and new_fairness<self.best_fairness:
            self.best_fairness = new_fairness
            self.best_performance = new_performance
            self.best_scales = action.copy()
        print("reward", reward)

        self.current_fairness = new_fairness
        self.current_performance = new_performance

        observation = np.concatenate([
            self.current_scales,
            [self.current_fairness, self.current_performance]
        ])

        info = {
            'fairness': new_fairness,
            'performance': new_performance,
            'best_fairness': self.best_fairness,
            'best_performance': self.best_performance,
            'best_scales': self.best_scales
        }

        return observation, reward, False, info
