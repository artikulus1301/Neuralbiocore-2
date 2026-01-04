import matplotlib.pyplot as plt
import numpy as np
import torch
from theory_of_mind import AgentSimulator

def run_tom_experiment():
    print("=== ЗАПУСК МУЛЬТИ-АГЕНТНОЙ СИМУЛЯЦИИ ToM (Section 19) ===")
    
    # 1. Инициализация агентов
    agent_A = AgentSimulator("Agent A")
    agent_B = AgentSimulator("Agent B")
    
    # Настройка начальных характеров
    # Агент А: Доверчивый (High Oxytocin equivalent -> low NE, high bias in M1)
    agent_A.social.M1_other.intentions += 0.5 # Априори верит в хорошее
    agent_A.chemistry.norepinephrine = torch.tensor(0.2)
    
    # Агент Б: Подозрительный / Тревожный
    agent_B.social.M1_other.intentions -= 0.2 # Априори ждет подвоха
    agent_B.chemistry.norepinephrine = torch.tensor(0.7)
    
    # Параметры симуляции
    dt = 0.1
    rounds = 50
    steps_per_round = 10 # Время на "обдумывание" между ходами
    
    # Логирование
    history = {
        'A_action': [], 'B_action': [],
        'A_trust_B': [], 'B_trust_A': [], # M1 beliefs
        'A_valence': [], 'B_valence': []
    }
    
    print(f"Starting interaction: {agent_A.name} (Trusting) vs {agent_B.name} (Anxious)")
    
    for r in range(rounds):
        # --- Фаза 1: Обдумывание (Internal Simulation) ---
        for _ in range(steps_per_round):
            agent_A.step_internal(dt)
            agent_B.step_internal(dt)
            
        # --- Фаза 2: Действие ---
        # Агенты принимают решение на основе текущих M1 и M0
        act_A = agent_A.decide_action()
        act_B = agent_B.decide_action()
        
        # --- Фаза 3: Восприятие и ToM Update ---
        # Агент А видит действие Б
        agent_A.perceive_social_signal(act_B, dt)
        agent_A.social.update_M2_recursion(torch.tensor([act_A]*10)) # "Я сделал X, он видел"
        
        # Агент Б видит действие А
        agent_B.perceive_social_signal(act_A, dt)
        agent_B.social.update_M2_recursion(torch.tensor([act_B]*10))
        
        # --- Фаза 4: Выплата наград (Game Payoff Matrix) ---
        # Классическая дилемма заключенного
        # Both Coop (>0): +0.5 each
        # Both Defect (<0): -0.2 each
        # One Coop, One Defect: Defector gets +1.0, Cooperator gets -0.5
        
        reward_A = 0
        reward_B = 0
        
        if act_A > 0 and act_B > 0:
            reward_A, reward_B = 0.5, 0.5 # Win-Win
        elif act_A < 0 and act_B < 0:
            reward_A, reward_B = -0.2, -0.2 # Lose-Lose
        elif act_A > 0 and act_B < 0:
            reward_A, reward_B = -0.5, 1.0 # A is sucker, B exploits
        elif act_A < 0 and act_B > 0:
            reward_A, reward_B = 1.0, -0.5 # A exploits, B is sucker
            
        agent_A.apply_reward(reward_A, dt)
        agent_B.apply_reward(reward_B, dt)
        
        # Логирование
        history['A_action'].append(act_A)
        history['B_action'].append(act_B)
        # M1 intentions - это вектор, берем среднее как уровень доверия
        history['A_trust_B'].append(torch.mean(agent_A.social.M1_other.intentions).item())
        history['B_trust_A'].append(torch.mean(agent_B.social.M1_other.intentions).item())
        history['A_valence'].append(agent_A.affect.valence.item())
        history['B_valence'].append(agent_B.affect.valence.item())
        
        if r % 10 == 0:
            print(f"Round {r}: A_act={act_A:.2f} B_act={act_B:.2f} | "
                  f"Trust A->B: {history['A_trust_B'][-1]:.2f}")

    # --- Визуализация ---
    rounds_arr = np.arange(rounds)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # График 1: Действия
    ax = axes[0]
    ax.plot(rounds_arr, history['A_action'], label='Agent A Action (Trusting)', color='blue', marker='o')
    ax.plot(rounds_arr, history['B_action'], label='Agent B Action (Anxious)', color='red', marker='x')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Behavior Dynamics: Cooperation (>0) vs Defection (<0)")
    ax.set_ylabel("Action")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 2: Модель Другого (M1 - Trust)
    ax = axes[1]
    ax.plot(rounds_arr, history['A_trust_B'], label="A's belief about B (M1)", color='cyan')
    ax.plot(rounds_arr, history['B_trust_A'], label="B's belief about A (M1)", color='orange')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Theory of Mind Dynamics: Inferred Intentions (M1)")
    ax.set_ylabel("Estimated Trustworthiness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 3: Эмоциональное состояние
    ax = axes[2]
    ax.plot(rounds_arr, history['A_valence'], label="Agent A Valence", color='blue', linestyle=':')
    ax.plot(rounds_arr, history['B_valence'], label="Agent B Valence", color='red', linestyle=':')
    ax.set_title("Affective State")
    ax.set_ylabel("Valence (-1 to 1)")
    ax.set_xlabel("Game Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tom_experiment()