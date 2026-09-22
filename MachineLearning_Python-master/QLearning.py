import numpy as np  # 导入 NumPy，用于数值计算和随机数生成
import matplotlib  # 导入 Matplotlib，用于可视化
matplotlib.use("Agg")  # 使用无界面后端，保证在没有图形环境的服务器上也能运行
import matplotlib.pyplot as plt  # 导入绘图模块


class GridWorld:  # 定义网格世界环境类
    """4x4 网格环境：S 是起点，X 是陷阱，T 是宝藏。"""

    def __init__(self, size=4):  # 构造函数，默认 4x4 网格
        self.size = size  # 网格边长（size x size 个格子）
        self.start = (0, 0)  # 起点坐标（左上角）
        self.treasure = (3, 3)  # 宝藏坐标（右下角）
        self.trap = (1, 1)  # 陷阱坐标
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 动作列表：上、右、下、左

    def reset(self):  # 每轮训练开始时重置环境
        return self.start  # 返回起始状态

    def step(self, state, action):  # 执行动作，返回新状态、奖励和是否结束
        row, col = state  # 解包当前状态的行列坐标
        row = int(np.clip(row + self.actions[action][0], 0, self.size - 1))  # 行坐标加上位移，裁剪到网格边界内
        col = int(np.clip(col + self.actions[action][1], 0, self.size - 1))  # 列坐标加上位移，裁剪到网格边界内
        next_state = (row, col)  # 组合成新状态的坐标

        if next_state == self.treasure:  # 走到宝藏：拿到大奖励，本轮结束
            return next_state, 100.0, True
        if next_state == self.trap:  # 走到陷阱：拿到大惩罚，本轮结束
            return next_state, -50.0, True
        return next_state, -0.1, False  # 普通格子：小惩罚，鼓励少走弯路


class QLearningAgent:  # 定义 Q-Learning 智能体类
    """带 ε-greedy 探索的表格型 Q-Learning 智能体。"""

    def __init__(self, env, n_states=16, n_actions=4, alpha=0.1,  # 构造函数，接收环境和超参数
                 gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.99, rng=None):
        self.env = env  # 保存环境引用
        self.n_states = n_states  # 状态总数（16 = 4x4）
        self.n_actions = n_actions  # 动作总数（上下左右 4 个）
        self.alpha = alpha  # 学习率，控制每次更新的幅度
        self.gamma = gamma  # 折扣因子，决定未来奖励的重要程度
        self.epsilon = epsilon  # 初始探索概率（1.0 表示一开始完全随机探索）
        self.epsilon_min = epsilon_min  # 探索概率的下限，保证后期仍有少量探索
        self.epsilon_decay = epsilon_decay  # 每轮结束后探索概率的衰减系数
        self.rng = rng if rng is not None else np.random.default_rng(42)  # 随机数生成器，固定种子保证结果可复现
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)  # Q 表：16 个状态 x 4 个动作，初始全零

    def state_index(self, state):  # 把 (行, 列) 坐标转成一维索引，方便查 Q 表
        row, col = state  # 解包坐标
        return row * self.env.size + col  # 二维转一维：索引 = 行 * 每行格子数 + 列

    def select_action(self, state):  # 根据当前状态选择动作（ε-greedy 策略）
        # ε-greedy：以 ε 概率探索，否则利用当前 Q 值最大的动作
        if self.rng.random() < self.epsilon:  # 以 ε 概率随机探索
            return int(self.rng.integers(self.n_actions))  # 随机返回一个动作编号
        state_idx = self.state_index(state)  # 把坐标转成 Q 表的行索引
        return int(np.argmax(self.q_table[state_idx]))  # 返回 Q 值最大的动作（利用）

    def update(self, state, action, reward, next_state, done):  # 核心：按 TD 规则更新 Q 值
        state_idx = self.state_index(state)  # 当前状态的 Q 表行索引
        next_state_idx = self.state_index(next_state)  # 新状态的 Q 表行索引

        if done:  # 如果本轮结束（宝藏或陷阱），没有未来价值
            target = reward  # 目标值就是即时奖励
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])  # 即时奖励 + 折扣后的新状态最大 Q 值

        td_error = target - self.q_table[state_idx, action]  # TD 误差：实际观察到的价值 - 当前估计值
        self.q_table[state_idx, action] += self.alpha * td_error  # 按学习率把 Q 值往目标方向调整

    def decay_epsilon(self):  # 每轮结束后降低探索概率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # 衰减但不低于下限

    def train(self, episodes=500, max_steps=100):  # 主训练循环
        rewards = []  # 记录每轮的累积奖励，用于观察收敛

        for _ in range(episodes):  # 训练 500 轮（episode = 从起点到终点的一局）
            state = self.env.reset()  # 每轮开始时回到起点
            episode_reward = 0.0  # 本轮累积奖励清零

            for _ in range(max_steps):  # 单轮最多走 100 步，防止死循环
                action = self.select_action(state)  # 用 ε-greedy 选一个动作
                next_state, reward, done = self.env.step(state, action)  # 执行动作，环境返回反馈
                self.update(state, action, reward, next_state, done)  # 用 TD 规则更新 Q 表
                episode_reward += reward  # 累加本轮奖励
                state = next_state  # 移动到新状态

                if done:  # 到达宝藏或陷阱，本轮结束
                    break  # 跳出内层循环

            self.decay_epsilon()  # 每轮结束后降低探索概率
            rewards.append(episode_reward)  # 记录本轮总奖励

        return rewards  # 返回所有轮次的奖励列表

    def best_policy(self):  # 训练结束后，从 Q 表提取每个格子的最优动作
        policy = np.empty((self.env.size, self.env.size), dtype="<U1")  # 创建一个和网格同尺寸的字符矩阵
        symbols = ["U", "R", "D", "L"]  # 动作编号 0-3 对应的方向符号

        for row in range(self.env.size):  # 遍历每一行
            for col in range(self.env.size):  # 遍历每一列
                if (row, col) == self.env.treasure:  # 宝藏格子标记为 T
                    policy[row, col] = "T"
                elif (row, col) == self.env.trap:  # 陷阱格子标记为 X
                    policy[row, col] = "X"
                else:
                    state_idx = self.state_index((row, col))  # 把坐标转成 Q 表索引
                    policy[row, col] = symbols[int(np.argmax(self.q_table[state_idx]))]  # 取 Q 值最大的动作作为该格子的方向

        return policy  # 返回完整的策略矩阵


def main():  # 程序入口
    env = GridWorld()  # 创建 4x4 网格环境
    agent = QLearningAgent(env)  # 创建 Q-Learning 智能体
    rewards = agent.train(episodes=500)  # 训练 500 轮

    print("学到的最优策略：")  # 打印标题
    print(agent.best_policy())  # 打印学到的策略矩阵
    print(f"\n最后 50 轮平均奖励：{np.mean(rewards[-50:]):.2f}")  # 打印后期平均奖励，评估收敛效果
    visualize(agent, rewards, "images/QLearning_01.png")  # 保存可视化结果


def visualize(agent, rewards, output_path):
    """绘制最大 Q 值热力图、策略箭头和奖励收敛曲线。"""
    size = agent.env.size
    max_q = np.max(agent.q_table, axis=1).reshape(size, size)

    fig, (ax_policy, ax_reward) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左图：最大 Q 值热力图；颜色越亮，表示这个状态对智能体的价值越高
    im = ax_policy.imshow(max_q, cmap="viridis", origin="upper")
    fig.colorbar(im, ax=ax_policy, label="Max Q-value")
    ax_policy.set_title("Q-Learning Value and Policy")
    ax_policy.set_xlabel("Column")
    ax_policy.set_ylabel("Row")
    ax_policy.set_xticks(np.arange(size))
    ax_policy.set_yticks(np.arange(size))

    # 在每个格子上画最优动作箭头，并标出宝藏和陷阱
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for row in range(size):
        for col in range(size):
            center = (col, row)
            if (row, col) == agent.env.treasure:
                ax_policy.text(col, row, "T", ha="center", va="center",
                               color="white", fontweight="bold")
            elif (row, col) == agent.env.trap:
                ax_policy.text(col, row, "X", ha="center", va="center",
                               color="white", fontweight="bold")
            else:
                state_idx = agent.state_index((row, col))
                action = int(np.argmax(agent.q_table[state_idx]))
                dr, dc = directions[action]
                # 箭头方向符合坐标系统：行向下增长，列向右增长
                ax_policy.annotate("", xy=(center[0] + dc * 0.35, center[1] + dr * 0.35),
                                   xytext=(center[0] - dc * 0.35, center[1] - dr * 0.35),
                                   arrowprops=dict(arrowstyle="->", color="white", lw=2))

    # 右图：每轮累积奖励曲线，前段波动大表示还在探索，后段稳定表示策略已收敛
    smooth_rewards = np.convolve(rewards, np.ones(20) / 20, mode="valid")
    ax_reward.plot(rewards, alpha=0.35, label="Episode reward")
    ax_reward.plot(smooth_rewards, linewidth=2, label="Moving average (20)")
    ax_reward.set_title("Training Reward Convergence")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Total reward")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")  # 打印图片保存位置


if __name__ == "__main__":  # 只有直接运行此文件时才执行 main()
    main()
