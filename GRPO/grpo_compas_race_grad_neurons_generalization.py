import torch
import torch.nn as nn
from models import CompasModel
from torch.distributions import Normal
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from Environment import NeuronScalingEnv
from utils import compute_metrics, convert_to_builtin_types, preprocess_compas, compute_neuron_forward_impact
import json
import os
import random

device = "cpu"


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, scale_bounds=(-1.0, 1.0)):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.scale_bounds = scale_bounds

    def forward(self, x):
       # x = self.relu(self.fc1(x))
       # x = self.relu(self.fc2(x))
       # x = self.relu(self.fc3(x))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
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
    """
    从单个环境采样一条轨迹（这个轨迹就是单步缩放）
    env: 神经元缩放环境
    policy_net: 策略网络

    返回：state, action, log_prob, reward
    """
    state = env.reset()  # 重置环境

    # 将状态转换为张量
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # 选择动作
    action_tensor, log_prob = policy_net.select_action(state_tensor)
    action = action_tensor.detach().cpu().numpy()[0]

    # 执行动作
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
    """
    收集多条轨迹作为一个批次
    """
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

    # 将列表转换为numpy数组
    batch_trajectories["states"] = np.array(batch_trajectories["states"])
    batch_trajectories["actions"] = np.array(batch_trajectories["actions"])
    batch_trajectories["log_probs"] = np.array(batch_trajectories["log_probs"])
    batch_trajectories["rewards"] = np.array(batch_trajectories["rewards"])

    # 计算信息的平均值
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
    """从轨迹中提取奖励，并标准化"""
    rewards = trajectories["rewards"]  # 提取奖励
    mean_reward = np.mean(rewards)  # 计算平均值
    std_reward = np.std(rewards) + 1e-8  # 计算标准差（1e-8是防止0除）
    advantages = (rewards - mean_reward) / std_reward  # 标准化

    return advantages


def grpo_update(trajectories, policy_net, optimizer, device="cpu", n_iterations=20, eps=0.2):
    # 计算标准化后的优势
    advantages = calc_advantages_with_grpo(trajectories)

    # 将数据转换为张量并移动到设备
    states = torch.tensor(trajectories["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(trajectories["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(trajectories["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(-1)

    batch_size = len(states)

    # 执行n_iterations次更新
    for _ in range(n_iterations):
        loss = 0

        for i in range(batch_size):
            state = states[i].unsqueeze(0)
            action = actions[i].unsqueeze(0)
            old_log_prob = old_log_probs[i]
            advantage = advantages[i]

            # 重新评估动作的对数概率
            dist = policy_net.get_distribution(state)
            new_log_prob = dist.log_prob(action).sum(dim=-1)

            # 计算概率比
            ratio = torch.exp(new_log_prob - old_log_prob)

            # 计算两个surrogate项
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

            # 计算损失
            trajectory_loss = -torch.min(surr1, surr2)
            loss += trajectory_loss

        # 用batch_size归一化损失
        loss /= batch_size

        # 更新策略网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def train_grpo(env, max_time_minutes=None, max_episodes=None, batch_size=10, lr=0.002, n_iterations=10, eps=0.2, policy_net=None):
    # 获取观测和动作维度
    obs_dim = len(env.reset())
    action_dim = len(env.candidate_neurons)
    hidden_dim = action_dim

    # 创建策略网络和优化器
    if policy_net is None:
        policy_net = PolicyNet(obs_dim, action_dim, hidden_dim).to(device)
    else:
        policy_net = policy_net.to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-3)

    # 记录训练历史
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
            # 检查退出条件
            if max_episodes is not None and episode >= max_episodes:
                break
            if max_time_minutes is not None and (time.time() - start_time) > max_time_minutes * 60:
                break
            # 收集一批轨迹
            trajectories, avg_reward, info = collect_batch_trajectories(env, policy_net, batch_size, device)

            # 使用GRPO更新策略
            loss = grpo_update(trajectories, policy_net, optimizer, device, n_iterations, eps)

            # 记录历史
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

    return env.best_fairness, env.best_performance, env.best_scales, total_steps, history["best_fairness"], policy_net


def plot_training_curves(history):
    """绘制训练曲线"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 奖励曲线
    axs[0, 0].plot(history["episode_rewards"])
    axs[0, 0].set_title("avg_rewards")
    axs[0, 0].set_xlabel("epoch")
    axs[0, 0].set_ylabel("reward")
    axs[0, 0].grid(True)

    # 公平性曲线
    axs[0, 1].plot(history["fairness"], label="current fairness")
    axs[0, 1].plot(history["best_fairness"], label="best fairness", linestyle="--")
    axs[0, 1].set_title("fairness metric")
    axs[0, 1].set_xlabel("epoch")
    axs[0, 1].set_ylabel("fairness")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 性能曲线
    axs[1, 0].plot(history["performance"], label="current performance")
    axs[1, 0].plot(history["best_performance"], label="best performance", linestyle="--")
    axs[1, 0].set_title("performance metric")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_ylabel("performance")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 公平性与性能的散点图
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
    hidden_dims = [64, 32, 16, 8, 4]
    key_layers_num = 2
    max_episodes = 200
    max_time_minutes = 5
    layers_mode = "key_layers"
    neurons_mode = "grad_neurons"
    model_paths = [
        'compas-0.9652-923728345-mms-0.1.pt',
        'compas-0.9688-11039402-mms-0.1.pt',
        'compas-0.9700-813498128-mms-0.1.pt',

        'compas-0.9737-4238509023-mms-0.1.pt',
        'compas-0.9589-347912312-mms-0.1.pt',
        'compas-0.9636-22349721-mms-0.1.pt',
        'compas-0.9670-3759631555-mms-0.1.pt',
        'compas-0.9669-2329670628-mms-0.1.pt',
        'compas-0.9664-1994190001-mms-0.1.pt',

        'compas-0.9712-563530960-mms-0.1.pt',
    ]
    key_neurons_num = {
        923728345: 47,
        11039402: 47,
        813498128: 52,
        4238509023: 56,
        347912312: 45,
        22349721: 45,
        3759631555: 49,
        2329670628: 53,
        1994190001: 53,
        563530960: 53
    }
    for model_path in model_paths:
        _, _, seed, _, dropout_rate = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout_rate)
        dataset = "none"
        attr = "race"

        _, X_val, X_test, _, y_val, y_test, sens_idx = preprocess_compas(seed, attr)
        sens_val = X_val[:, sens_idx]
        sens_test = X_test[:, sens_idx]
        sens_classes = [0, 1]

        model = CompasModel(p=p).to(device)
        state_dict = torch.load(f"../saved_models/compas/{model_path}", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load(f"../saved_models/adult/{model_path}"))
        model.eval()

        impact_tuples_sorted = compute_neuron_forward_impact(model, X_val, key_layers_num=3)
        print(impact_tuples_sorted)

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

        # 需要获取key neurons
        top_k = key_neurons_num[seed]
        key_neurons = [(layer_name, neuron_index) for layer_name, neuron_index, _ in impact_tuples_sorted[:top_k]]
        print("key neurons num:", len(key_neurons))
        print("key neurons:", key_neurons)

        save_dict = {
            "key_neurons": key_neurons,
            "key_neurons num": len(key_neurons),
            "baseline_fairness_eod on Val": baseline_eod,
            "baseline_performance on Val": baseline_f1,
            "baseline_accuracy on Val": baseline_acc,
            "baseline_fairness_eod on Test": baseline_eod_test,
            "baseline_performance on Test": baseline_f1_test,
            "baseline_accuracy on Test": baseline_acc_test,
        }

        save_dict = convert_to_builtin_types(save_dict)

        save_dir = f"intermediate results/Compas_race"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{seed}_{layers_mode}_{neurons_mode}_{max_time_minutes}.json")
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))

        env = NeuronScalingEnv(model, key_neurons, baseline_eod, baseline_f1,
                               X_val, y_val, sens_val, sens_classes, dataset)

        start_time = time.time()

        best_fairness, best_performance, best_scales, total_steps, best_fairness_list, policy_net = train_grpo(
            env=env,
            max_time_minutes=max_time_minutes,
            batch_size=10,
            lr=0.005,
            n_iterations=10,
            eps=0.2
        )
        torch.save(policy_net.state_dict(), "policy_net_compas_race_val.pth")

        optimization_time = time.time() - start_time
        print("Optimization Time Cost:", optimization_time)

        save_dict["repaired_fairness_eod on Val"] = best_fairness
        save_dict["repaired_performance on val"] = best_performance

        model.eval()
        model.register_scaling_hooks(key_neurons, best_scales)
        repaired_eod, repaired_dpd, repaired_di, repaired_f1, repaired_acc = compute_metrics(
            model, X_val, y_val, sens_val, sens_classes, dataset
        )
        repaired_eod_test, repaired_dpd_test, repaired_di_test, repaired_f1_test, repaired_acc_test = compute_metrics(
            model, X_test, y_test, sens_test, sens_classes, dataset
        )

        save_dict["repaired_accuracy on val"] = repaired_acc
        save_dict["repaired_fairness_eod on Test (Before fine_tune)"] = repaired_eod_test
        save_dict["repaired_performance on Test (Before fine_tune)"] = repaired_f1_test
        save_dict["repaired_accuracy on Test (Before fine_tune)"] = repaired_acc_test
        save_dict["optimization time cost"] = optimization_time
        save_dict["key_neurons_final_number"] = len(key_neurons)

        save_dict["repaired_fairness_eod on Test (After fine_tune)"] = repaired_eod_test
        save_dict["repaired_performance on Test (After fine_tune)"] = repaired_f1_test
        save_dict["repaired_accuracy on Test (After fine_tune)"] = repaired_acc_test
        save_dict = convert_to_builtin_types(save_dict)

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))
