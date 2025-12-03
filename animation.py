# animal.py
# ============================================
#  本文件包含：
#  1. 井字棋 TD-Learning 智能体训练代码
#  2. 训练后两个 Agent 的对战动画
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# -----------------------------
#         Agent 类
# -----------------------------
class Agent():

    def __init__(self, OOXX_Index, Epsilon, Alpha):
        self.index = OOXX_Index
        self.epsilon = Epsilon
        self.alpha = Alpha
        self.value = np.zeros((3,3,3,3,3,3,3,3,3))
        self.stored_Outcome = np.zeros(9).astype(np.int8)

    def reset(self):
        self.stored_Outcome = np.zeros(9).astype(np.int8)

    def move(self, State):
        Outcome = State.copy()
        available = np.where(Outcome == 0)[0]

        # ε-Greedy
        if np.random.binomial(1, self.epsilon):
            Outcome[np.random.choice(available)] = self.index
        else:
            temp_Value = np.zeros(len(available))
            for i in range(len(available)):
                temp_Outcome = Outcome.copy()
                temp_Outcome[available[i]] = self.index
                temp_Value[i] = self.value[tuple(temp_Outcome)]
            choose = np.argmax(temp_Value)
            Outcome[available[choose]] = self.index

        # TD Error
        Error = self.value[tuple(Outcome)] - self.value[tuple(self.stored_Outcome)]
        self.value[tuple(self.stored_Outcome)] += self.alpha * Error
        self.stored_Outcome = Outcome.copy()

        return Outcome


# -----------------------------
#        判断输赢
# -----------------------------
def Judge(Outcome, curPlayer):
    Triple = np.repeat(curPlayer.index, 3)
    winner = 0
    if 0 not in Outcome:
        winner = 3  # 平局

    # 三行
    if (Outcome[0:3]==Triple).all() or (Outcome[3:6]==Triple).all() or (Outcome[6:9]==Triple).all():
        winner = curPlayer.index
    # 三列
    if (Outcome[0:7:3]==Triple).all() or (Outcome[1:8:3]==Triple).all() or (Outcome[2:9:3]==Triple).all():
        winner = curPlayer.index
    # 对角线
    if (Outcome[0:9:4]==Triple).all() or (Outcome[2:7:2]==Triple).all():
        winner = curPlayer.index

    return winner


# -------------------------------------
#         动画：最终对战可视化
# -------------------------------------
def simulate_animation(agent1, agent2, save_name="final_match.gif"):
    # 强制使用贪心策略
    agent1.epsilon = 0
    agent2.epsilon = 0
    agent1.reset()
    agent2.reset()

    State = np.zeros(9).astype(np.int8)
    curPlayer = agent1
    opponent = agent2
    game_states = [State.copy()]

    winner = 0
    while winner == 0:
        Outcome = curPlayer.move(State)
        winner = Judge(Outcome, curPlayer)
        game_states.append(Outcome.copy())

        State = Outcome
        curPlayer, opponent = opponent, curPlayer

    # -----------------
    # 绘制动画
    # -----------------
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Final Match Simulation")

    def draw_board(board):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

        # 格子
        for i in range(1,3):
            ax.axhline(i, color="black")
            ax.axvline(i, color="black")

        # 落子
        for idx, v in enumerate(board):
            r, c = divmod(idx,3)
            if v == 1:
                ax.text(c+0.5, 2.5-r, "O", ha="center", va="center", fontsize=40, color="blue")
            elif v == 2:
                ax.text(c+0.5, 2.5-r, "X", ha="center", va="center", fontsize=40, color="red")

        ax.set_xlim(0,3)
        ax.set_ylim(0,3)

    def update(frame):
        draw_board(game_states[frame])

    ani = animation.FuncAnimation(fig, update, frames=len(game_states), interval=400)

    # -----------------
    # 保存 GIF
    # -----------------
    print("\n正在保存 GIF ...")
    ani.save(save_name, writer="pillow", fps=2)
    print(f"GIF 已保存为： {save_name}")

    plt.show()

    # 打印胜负
    print("\n最终结果 winner =", winner)
    if winner == 1:
        print("Agent1 (O) 获胜")
    elif winner == 2:
        print("Agent2 (X) 获胜")
    else:
        print("平局")

    return ani


# -------------------------------------
#        主程序：训练 + 动画
# -------------------------------------
def main():

    Agent1 = Agent(1, 0.1, 0.1)
    Agent2 = Agent(2, 0.1, 0.1)

    Trial = 30000
    Winner = np.zeros(Trial)

    for i in range(Trial):
        if i == 20000:
            Agent1.epsilon = 0
            Agent2.epsilon = 0
        Agent1.reset()
        Agent2.reset()
        State = np.zeros(9).astype(np.int8)
        winner = 0

        curPlayer = Agent1
        opponent = Agent2

        while winner == 0:
            Outcome = curPlayer.move(State)
            winner = Judge(Outcome, curPlayer)
            if winner == curPlayer.index:
                curPlayer.value[tuple(Outcome)] = 1
                opponent.value[tuple(State)] = -1

            curPlayer, opponent = opponent, curPlayer
            State = Outcome

        Winner[i] = winner

    print("训练完成！开始播放最终对战动画…")
    simulate_animation(Agent1, Agent2)


if __name__ == "__main__":
    main()
