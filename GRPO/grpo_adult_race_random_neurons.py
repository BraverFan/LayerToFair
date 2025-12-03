import random

import torch
import torch.nn as nn
from models import AdultCensusModel
from torch.distributions import Normal
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from Environment import NeuronScalingEnv
from utils import compute_metrics, convert_to_builtin_types, preprocess_adult_census
from KeyNeuronsIdentification import KeyNeuronsIdentification
import json
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, scale_bounds=(-1.0, 1.0)):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.scale_bounds = scale_bounds

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        mean = self.tanh(self.fc_mean(x))
        return mean

    def get_distribution(self, obs):
        mean = self.forward(obs)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def select_action(self, obs):
        with torch.no_grad():
            dist = self.get_distribution(obs)
            action = dist.sample()
            action = torch.clamp(action, self.scale_bounds[0], self.scale_bounds[1])
        return action, dist.log_prob(action).sum(dim=-1)


def collect_trajectory(env, policy_net, device="cpu"):
    state = env.reset()  

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    action_tensor, log_prob = policy_net.select_action(state_tensor)
    action = action_tensor.detach().cpu().numpy()[0]

    next_state, reward, done, info = env.step(action)

    trajectory = {
        "state": state,
        "action": action,
        "log_prob": log_prob.item(),
        "reward": reward,
        "info": info
    }

    return trajectory


def collect_batch_trajectories(env, policy_net, batch_size=10, device="cpu"):
    batch_trajectories = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "rewards": []
    }

    fairness_list = []
    performance_list = []
    reward_list = []

    for _ in range(batch_size):
        trajectory = collect_trajectory(env, policy_net, device)

        batch_trajectories["states"].append(trajectory["state"])
        batch_trajectories["actions"].append(trajectory["action"])
        batch_trajectories["log_probs"].append(trajectory["log_prob"])
        batch_trajectories["rewards"].append(trajectory["reward"])

        fairness_list.append(trajectory["info"]["fairness"])
        performance_list.append(trajectory["info"]["performance"])
        reward_list.append(trajectory["reward"])

    batch_trajectories["states"] = np.array(batch_trajectories["states"])
    batch_trajectories["actions"] = np.array(batch_trajectories["actions"])
    batch_trajectories["log_probs"] = np.array(batch_trajectories["log_probs"])
    batch_trajectories["rewards"] = np.array(batch_trajectories["rewards"])

    avg_fairness = np.mean(fairness_list)
    avg_performance = np.mean(performance_list)
    avg_reward = np.mean(reward_list)

    info = {
        "avg_fairness": avg_fairness,
        "avg_performance": avg_performance,
        "best_fairness": env.best_fairness,
        "best_performance": env.best_performance,
        "best_scales": env.best_scales
    }

    return batch_trajectories, avg_reward, info


def calc_advantages_with_grpo(trajectories):
    rewards = trajectories["rewards"]  
    mean_reward = np.mean(rewards)  
    std_reward = np.std(rewards) + 1e-8  
    advantages = (rewards - mean_reward) / std_reward  

    return advantages


def grpo_update(trajectories, policy_net, optimizer, device="cpu", n_iterations=20, eps=0.2):
    advantages = calc_advantages_with_grpo(trajectories)

    states = torch.tensor(trajectories["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(trajectories["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(trajectories["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(-1)

    batch_size = len(states)

    for _ in range(n_iterations):
        loss = 0

        for i in range(batch_size):
            state = states[i].unsqueeze(0)
            action = actions[i].unsqueeze(0)
            old_log_prob = old_log_probs[i]
            advantage = advantages[i]

            dist = policy_net.get_distribution(state)
            new_log_prob = dist.log_prob(action).sum(dim=-1)

            ratio = torch.exp(new_log_prob - old_log_prob)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

            trajectory_loss = -torch.min(surr1, surr2)
            loss += trajectory_loss

        loss /= batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def train_grpo(env, max_time_minutes=5, batch_size=10, lr=0.002, n_iterations=10, eps=0.2):
    obs_dim = len(env.reset())
    action_dim = len(env.candidate_neurons)
    hidden_dim = action_dim

    policy_net = PolicyNet(obs_dim, action_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    history = {
        "episode_rewards": [],
        "fairness": [],
        "performance": [],
        "best_fairness": [],
        "best_performance": [],
        "episodes_completed": 0
    }

    start_time = time.time()
    end_time = start_time + max_time_minutes * 60
    episode = 0

    with tqdm() as pbar:
        while time.time() < end_time:
            trajectories, avg_reward, info = collect_batch_trajectories(env, policy_net, batch_size, device)

            loss = grpo_update(trajectories, policy_net, optimizer, device, n_iterations, eps)

            history["episode_rewards"].append(avg_reward)
            history["fairness"].append(info["avg_fairness"])
            history["performance"].append(info["avg_performance"])
            history["best_fairness"].append(info["best_fairness"])
            history["best_performance"].append(info["best_performance"])
            history["episodes_completed"] += 1

            remaining_time = max(0, end_time - time.time())
            pbar.set_description(
                f"Episode {episode}, Remaining: {remaining_time:.1f}s")
            pbar.update(1)
            episode += 1

            if episode % 10 == 0 or episode == max_episodes - 1:
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Fairness: {info['avg_fairness']:.4f}, "
                      f"Performance: {info['avg_performance']:.4f}, Best Fairness: {info['best_fairness']:.4f}, "
                      f"Best Performance: {info['best_performance']:.4f}")

    print("\n最佳结果:")
    print(f"公平性: {env.best_fairness:.4f}")
    print(f"性能: {env.best_performance:.4f}")
    print(f"最佳缩放因子: {env.best_scales}")

    plot_training_curves(history)

    return env.best_fairness, env.best_performance, env.best_scales, history["episodes_completed"] * batch_size


def plot_training_curves(history):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(history["episode_rewards"])
    axs[0, 0].set_title("avg_rewards")
    axs[0, 0].set_xlabel("epoch")
    axs[0, 0].set_ylabel("reward")
    axs[0, 0].grid(True)

    axs[0, 1].plot(history["fairness"], label="current fairness")
    axs[0, 1].plot(history["best_fairness"], label="best fairness", linestyle="--")
    axs[0, 1].set_title("fairness metric")
    axs[0, 1].set_xlabel("epoch")
    axs[0, 1].set_ylabel("fairness")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(history["performance"], label="current performance")
    axs[1, 0].plot(history["best_performance"], label="best performance", linestyle="--")
    axs[1, 0].set_title("performance metric")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_ylabel("performance")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].scatter(history["fairness"], history["performance"], alpha=0.5)
    axs[1, 1].scatter([history["best_fairness"][-1]], [history["best_performance"][-1]],
                      color="red", s=100, marker="*", label="best")
    axs[1, 1].set_title("fairness vs performance")
    axs[1, 1].set_xlabel("fairness")
    axs[1, 1].set_ylabel("performance")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hidden_dims = [64, 128, 256, 256, 256, 256, 128, 64]
    key_layers_num = 4
    max_episodes = 200
    max_time_minutes = 5
    layers_mode = "key_layers"
    neurons_mode = "random_neurons"
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

    key_neurons_num = {
        42: 91,
        74364756: 86,
        3872938: 79,
        586933: 97,
        93276487: 85,
        6489372: 99,
        2875882: 97,
        1894375983: 98,
        89540917: 99,
        582093843: 97
    }
    for model_path in model_paths:
        _, _, seed, _, dropout_rate = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout_rate)
        dataset = "adult_race"
        adult_race = (23, 27)

        _, X_val, _, y_val, sens_idx = preprocess_adult_census(seed)
        sens_val = X_val[:, adult_race[0]: adult_race[1]]
        sens_classes = [0, 1]

        model = AdultCensusModel(p=p).to(device)
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

        layer_names = [f"layer{i + 1}" for i in range(key_layers_num)]
        candidate_neurons = []
        for i, layer_name in enumerate(layer_names):
            num_neurons = hidden_dims[i]
            for neuron_index in range(num_neurons):
                candidate_neurons.append((layer_name, neuron_index))
        key_neurons = random.sample(candidate_neurons, key_neurons_num[seed])
        print("key neurons num:", len(key_neurons))
        print("key neurons:", key_neurons)

        save_dict = {
            "key_neurons": key_neurons,
            "baseline_fairness_eod on Val": baseline_eod,
            "baseline_fairness_dpd on Val": baseline_dpd,
            "baseline_fairness_di on Val": baseline_di,
            "baseline_performance on Val": baseline_f1,
            "baseline_accuracy on Val": baseline_acc,
        }

        save_dict = convert_to_builtin_types(save_dict)

        save_dir = f"Ablation experiment/Adult_race_random_neurons"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{seed}_{layers_mode}_{neurons_mode}_{max_time_minutes}.json")
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))

        env = NeuronScalingEnv(model, key_neurons, baseline_eod, baseline_f1,
                               X_val, y_val, sens_val, sens_classes, dataset)
        start_time = time.time()

        best_fairness, best_performance, best_scales, num_iterations = train_grpo(
            env=env,
            max_time_minutes=max_time_minutes,
            batch_size=10,
            lr=0.005,
            n_iterations=10,
            eps=0.2
        )

        optimization_time = time.time() - start_time
        print("Optimization Time Cost:", optimization_time)

        save_dict["repaired_fairness_eod on Val"] = best_fairness
        save_dict["repaired_performance on val"] = best_performance

        model.register_scaling_hooks(key_neurons, best_scales)
        repaired_eod, repaired_dpd, repaired_di, repaired_f1, repaired_acc = compute_metrics(
            model, X_val, y_val, sens_val, sens_classes, dataset
        )
        save_dict["repaired_accuracy on val"] = repaired_acc
        save_dict["repaired_fairness_eod on Test"] = repaired_eod_test
        save_dict["repaired_performance on Test"] = repaired_f1_test
        save_dict["repaired_accuracy on Test"] = repaired_acc_test
        save_dict["optimization time cost"] = optimization_time

        save_dict = convert_to_builtin_types(save_dict)

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))
