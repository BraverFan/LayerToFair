import torch
import torch.nn as nn
from models import BankModel
from torch.distributions import Normal
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from Environment import NeuronScalingEnv
from utils import compute_metrics, convert_to_builtin_types, preprocess_bank
from KeyNeuronsIdentification import KeyNeuronsIdentification
import json
import os

device = "cpu"

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, scale_bounds=(-1.0, 1.0)):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.dropout = nn.Dropout(p=0.2)

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


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        value = self.fc_value(x)
        return value


def collect_trajectory(env, policy_net, value_net, device="cpu"):
    state = env.reset()

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    action_tensor, log_prob = policy_net.select_action(state_tensor)
    action = action_tensor.detach().cpu().numpy()[0]

    with torch.no_grad():
        value = value_net(state_tensor).item()

    next_state, reward, done, info = env.step(action)

    trajectory = {
        "state": state,
        "action": action,
        "log_prob": log_prob.item(),
        "reward": reward,
        "value": value,
        "info": info
    }

    return trajectory


def collect_batch_trajectories(env, policy_net, value_net, batch_size=10, device="cpu"):
    batch_trajectories = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "values": []
    }

    fairness_list = []
    performance_list = []
    reward_list = []

    for _ in range(batch_size):
        trajectory = collect_trajectory(env, policy_net, value_net, device)

        batch_trajectories["states"].append(trajectory["state"])
        batch_trajectories["actions"].append(trajectory["action"])
        batch_trajectories["log_probs"].append(trajectory["log_prob"])
        batch_trajectories["rewards"].append(trajectory["reward"])
        batch_trajectories["values"].append(trajectory["value"])

        fairness_list.append(trajectory["info"]["fairness"])
        performance_list.append(trajectory["info"]["performance"])
        reward_list.append(trajectory["reward"])

    batch_trajectories["states"] = np.array(batch_trajectories["states"])
    batch_trajectories["actions"] = np.array(batch_trajectories["actions"])
    batch_trajectories["log_probs"] = np.array(batch_trajectories["log_probs"])
    batch_trajectories["rewards"] = np.array(batch_trajectories["rewards"])
    batch_trajectories["values"] = np.array(batch_trajectories["values"])

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


def compute_advantages_ppo(rewards, values, gamma=0.99, lam=0.95):
    rewards = np.array(rewards)
    values = np.array(values)

    next_values = np.zeros_like(values)

    deltas = rewards + gamma * next_values - values

    advantages = np.zeros_like(rewards)
    advantage = 0
    for t in reversed(range(len(rewards))):
        advantage = deltas[t] + gamma * lam * advantage
        advantages[t] = advantage

    returns = advantages + values

    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return advantages, returns


def ppo_update(trajectories, policy_net, value_net, policy_optimizer, value_optimizer,
               device="cpu", n_epochs=10, eps_clip=0.2, vf_coef=0.5, ent_coef=0.01):
    advantages, returns = compute_advantages_ppo(trajectories["rewards"], trajectories["values"])

    states = torch.tensor(trajectories["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(trajectories["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(trajectories["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    batch_size = len(states)

    for epoch in range(n_epochs):
        indices = torch.randperm(batch_size)

        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_entropy_loss = 0

        for i in indices:
            state = states[i].unsqueeze(0)
            action = actions[i].unsqueeze(0)
            old_log_prob = old_log_probs[i]
            advantage = advantages[i]
            return_val = returns[i]

            dist = policy_net.get_distribution(state)
            new_log_prob = dist.log_prob(action).sum(dim=-1)

            ratio = torch.exp(new_log_prob - old_log_prob)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2)

            entropy = dist.entropy().sum(dim=-1)
            entropy_loss = -ent_coef * entropy

            current_value = value_net(state).squeeze()
            value_loss = vf_coef * ((current_value - return_val) ** 2)

            total_loss = policy_loss + value_loss + entropy_loss

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()
            policy_optimizer.step()
            value_optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_entropy_loss += entropy_loss.item()

    avg_policy_loss = epoch_policy_loss / (n_epochs * batch_size)
    avg_value_loss = epoch_value_loss / (n_epochs * batch_size)
    avg_entropy_loss = epoch_entropy_loss / (n_epochs * batch_size)

    return avg_policy_loss + avg_value_loss + avg_entropy_loss


def train_ppo(env, max_time_minutes=None, max_episodes=None, batch_size=10, lr=0.002,
              n_epochs=10, eps_clip=0.2, policy_net=None, value_net=None):
    obs_dim = len(env.reset())
    action_dim = len(env.candidate_neurons)
    hidden_dim = action_dim

    if policy_net is None:
        policy_net = PolicyNet(obs_dim, action_dim, hidden_dim).to(device)
    else:
        policy_net = policy_net.to(device)

    if value_net is None:
        value_net = ValueNet(obs_dim, hidden_dim).to(device)
    else:
        value_net = value_net.to(device)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

    history = {
        "episode_rewards": [],
        "fairness": [],
        "performance": [],
        "best_fairness": [],
        "best_performance": []
    }

    start_time = time.time()
    episode = 0
    total_steps = 0

    with tqdm() as pbar:
        while True:
            if max_episodes is not None and episode >= max_episodes:
                break
            if max_time_minutes is not None and (time.time() - start_time) > max_time_minutes * 60:
                break

            trajectories, avg_reward, info = collect_batch_trajectories(env, policy_net, value_net, batch_size, device)

            loss = ppo_update(trajectories, policy_net, value_net, policy_optimizer, value_optimizer,
                              device, n_epochs, eps_clip)

            history["episode_rewards"].append(avg_reward)
            history["fairness"].append(info["avg_fairness"])
            history["performance"].append(info["avg_performance"])
            history["best_fairness"].append(info["best_fairness"])
            history["best_performance"].append(info["best_performance"])

            remaining_time = ""
            if max_time_minutes is not None:
                remaining_secs = max(0, (max_time_minutes * 60) - (time.time() - start_time))
                remaining_time = f"Remaining: {remaining_secs:.1f}s"

            pbar.set_description(f"Episode {episode}, {remaining_time}")
            pbar.update(1)

            if episode % 10 == 0:
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Fairness: {info['avg_fairness']:.4f}, "
                      f"Performance: {info['avg_performance']:.4f}, Best Fairness: {info['best_fairness']:.4f}, "
                      f"Best Performance: {info['best_performance']:.4f}")

            episode += 1
            total_steps += batch_size

    print(f"\n训练完成，用时 {time.time() - start_time:.2f}s，共 {episode} 轮")
    print(f"最佳公平性: {env.best_fairness:.4f}")
    print(f"最佳性能: {env.best_performance:.4f}")
    print(f"最佳缩放因子: {env.best_scales}")

    plot_training_curves(history)

    return env.best_fairness, env.best_performance, env.best_scales, total_steps, history[
        "best_fairness"], policy_net, value_net


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
    hidden_dims = [128, 128, 128, 128, 128, 128, 128]
    key_layers_num = 3
    threshold = 0.01
    max_episodes = 200
    max_time_minutes = 5
    layers_mode = "key_layers"
    neurons_mode = "key_neurons"
    model_paths = [
        'bank-0.5471-934827-ss-0.3.pt',
        'bank-0.5706-3989056-ss-0.3.pt',
        'bank-0.5741-385491921-ss-0.4.pt',
        'bank-0.5559-28939112-ss-0.3.pt',
        'bank-0.5568-98392834-ss-0.4.pt',
        'bank-0.5691-30048091-ss-0.4.pt',
        'bank-0.5760-289395393-ss-0.4.pt',
        'bank-0.5636-82932321-ss-0.4.pt',
        'bank-0.5679-66663820-ss-0.5.pt',
        'bank-0.5794-12002342-ss-0.3.pt',
    ]
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

    for model_path in model_paths:
        _, _, seed, _, dropout_rate = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout_rate)
        dataset = "none"

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_bank(seed)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = [sens_classes_all[0], sens_classes_all[3]]

        model = BankModel(p=p).to(device)
        state_dict = torch.load(f"../saved_models/bank/{model_path}", map_location=torch.device('cpu'))
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

        hyperparameter_search_type = "RandomSearch"  # RandomSearch or GridSearch
        start_time = time.time()
        layer_sizes_accuracy, key_neurons, total_train_time = (
            KeyNeuronsIdentification(model, hidden_dims, X_val,
                    sens_idx, threshold, hyperparameter_search_type).identify())
        identification_time = time.time() - start_time
        print("Identification Time Cost:", identification_time)

        save_dict = {
            "layer_sizes_accuracy": layer_sizes_accuracy,
            "key_neurons": key_neurons,
            "baseline_fairness_eod on Val": baseline_eod,
            "baseline_performance on Val": baseline_f1,
            "baseline_accuracy on Val": baseline_acc,
            "baseline_fairness_eod on Test": baseline_eod_test,
            "baseline_performance on Test": baseline_f1_test,
            "baseline_accuracy on Test": baseline_acc_test,
        }

        save_dict = convert_to_builtin_types(save_dict)

        save_dir = f"intermediate results/Bank"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{seed}_{layers_mode}_{neurons_mode}_{max_time_minutes}_ppo.json")
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))

        layer_name_keys = list(key_neurons.keys())[:key_layers_num]

        key_neurons_final = []
        for layer in layer_name_keys:
            key_neurons_final.extend((layer, i) for i in key_neurons[layer])
        print("key_neurons_final", key_neurons_final)

        env = NeuronScalingEnv(model, key_neurons_final, baseline_eod, baseline_f1,
                               X_val, y_val, sens_val, sens_classes, dataset)
        start_time = time.time()

        best_fairness, best_performance, best_scales, total_steps, best_fairness_list, policy_net, value_net = train_ppo(
            env=env,
            max_time_minutes=max_time_minutes,
            batch_size=10,
            lr=0.005,
            n_epochs=10,
            eps_clip=0.2
        )

        torch.save(policy_net.state_dict(), "policy_net_bank_val.pth")
        torch.save(value_net.state_dict(), "value_net_bank_val.pth")

        optimization_time = time.time() - start_time
        print("Optimization Time Cost:", optimization_time)

        save_dict["repaired_fairness_eod on Val"] = best_fairness
        save_dict["repaired_performance on Val"] = best_performance

        model.eval()
        model.register_scaling_hooks(key_neurons, best_scales)
        repaired_eod, repaired_dpd, repaired_di, repaired_f1, repaired_acc = compute_metrics(
            model, X_val, y_val, sens_val, sens_classes, dataset
        )
        repaired_eod_test, repaired_dpd_test, repaired_di_test, repaired_f1_test, repaired_acc_test = compute_metrics(
            model, X_test, y_test, sens_test, sens_classes, dataset
        )

        save_dict["repaired_accuracy on val"] = repaired_acc
        save_dict["repaired_fairness_eod on Test"] = repaired_eod_test
        save_dict["repaired_performance on Test"] = repaired_f1_test
        save_dict["repaired_accuracy on Test"] = repaired_acc_test
        save_dict["optimization time cost"] = optimization_time

        save_dict = convert_to_builtin_types(save_dict)

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))
