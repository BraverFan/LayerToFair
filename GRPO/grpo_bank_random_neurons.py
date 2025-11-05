import random

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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


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


def train_grpo(env, max_time_minutes=5, batch_size=10, lr=0.002, n_iterations=10, eps=0.2):
    """
    使用GRPO训练神经元缩放环境

    参数:
    env: 神经元缩放环境
    max_episodes: 最大训练轮数
    batch_size: 每批次采样的轨迹数
    lr: 学习率
    n_iterations: 每批次的更新迭代次数
    eps: GRPO的截断参数
    performance_threshold: 性能阈值，低于此值的解决方案不会被考虑
    early_stop_fairness: 如果达到此公平性指标，则提前停止训练

    返回:
    history: 训练历史记录
    """
    # 获取观测和动作维度
    obs_dim = len(env.reset())
    action_dim = len(env.candidate_neurons)
    hidden_dim = action_dim

    # 创建策略网络和优化器
    policy_net = PolicyNet(obs_dim, action_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # 记录训练历史
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

    # 主训练循环
    with tqdm() as pbar:
        while time.time() < end_time:
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
            history["episodes_completed"] += 1

            remaining_time = max(0, end_time - time.time())
            pbar.set_description(
                f"Episode {episode}, Remaining: {remaining_time:.1f}s")
            pbar.update(1)
            episode += 1

            # 打印信息
            if episode % 10 == 0 or episode == max_episodes - 1:
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Fairness: {info['avg_fairness']:.4f}, "
                      f"Performance: {info['avg_performance']:.4f}, Best Fairness: {info['best_fairness']:.4f}, "
                      f"Best Performance: {info['best_performance']:.4f}")

    # 打印最佳结果
    print("\n最佳结果:")
    print(f"公平性: {env.best_fairness:.4f}")
    print(f"性能: {env.best_performance:.4f}")
    print(f"最佳缩放因子: {env.best_scales}")

    # 绘制训练曲线
    plot_training_curves(history)

    return env.best_fairness, env.best_performance, env.best_scales, history["episodes_completed"] * batch_size


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
    hidden_dims = [128, 128, 128, 128, 128, 128, 128]
    key_layers_num = 4
    max_episodes = 200
    max_time_minutes = 5
    layers_mode = "key_layers"
    neurons_mode = "random_neurons"
    model_paths = [
        'bank-0.5418-92357462-ss-0.2.pt',
        'bank-0.5498-67802376-ss-0.2.pt',
        'bank-0.5371-48689224-ss-0.2.pt',
        'bank-0.5489-397895609-ss-0.2.pt',
        'bank-0.5453-298373984-ss-0.2.pt',
        'bank-0.5446-47857899-ss-0.2.pt',
        'bank-0.5612-638209430-ss-0.2.pt',
        'bank-0.5615-183496-ss-0.2.pt',
        'bank-0.5505-529892001-ss-0.2.pt',
        'bank-0.5580-823626012-ss-0.2.pt',
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

    key_neurons_num = {
        92357462: 108,
        67802376: 111,
        48689224: 104,
        397895609: 105,
        298373984: 106,
        47857899: 103,
        638209430: 104,
        183496: 108,
        529892001: 97,
        823626012: 102
    }

    for model_path in model_paths:
        _, _, seed, _, dropout_rate = model_path[:-3].split("-")
        seed = int(seed)
        p = float(dropout_rate)
        dataset = "none"

        _, X_val, _, y_val, sens_idx = preprocess_bank(seed)
        sens_val = X_val[:, sens_idx]
        sens_classes = [sens_classes_all[0], sens_classes_all[3]]

        model = BankModel(p=p).to(device)
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

        # 需要获取key neurons
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

        save_dir = f"Ablation experiment/Bank_random_neurons"
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

        model.eval()
        model.register_scaling_hooks(key_neurons, best_scales)
        repaired_eod, repaired_dpd, repaired_di, repaired_f1, repaired_acc = compute_metrics(
            model, X_val, y_val, sens_val, sens_classes, dataset
        )
        save_dict["repaired_fairness_dpd on val"] = repaired_dpd
        save_dict["repaired_fairness_di on val"] = repaired_di
        save_dict["repaired_accuracy on val"] = repaired_acc
        save_dict["optimization time cost"] = optimization_time
        save_dict["num_iteration"] = num_iterations

        save_dict = convert_to_builtin_types(save_dict)

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4, separators=(',', ': '))
