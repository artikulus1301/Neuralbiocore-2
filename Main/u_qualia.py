import torch
import torch.nn as nn
import numpy as np
import math
import zlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

# --- ПОПЫТКА ИМПОРТА БАЗОВОГО ЯДРА ---
# Если файла A_NeuralBiocore_vDeltha нет, создаем заглушки для автономной работы.
try:
    import Neuralbiocore_U_for_GPU as nb
except ImportError:
    class DummySimulator:
        def __init__(self):
            self.gwt = type('GWT', (), {'broadcast_signal': torch.zeros(100)})()
            self.affect = type('Affect', (), {
                'valence': torch.tensor(0.0),
                'arousal': torch.tensor(0.0),
                'insula_activity': torch.tensor(0.0),
                'amygdala_activity': torch.tensor(0.0)
            })()
            self.chemistry = type('Chemistry', (), {
                'norepinephrine': torch.tensor(0.0),
                'dopamine': torch.tensor(0.0),
                'propofol_conc': torch.tensor(0.0)
            })()
            self.simulation_time = 0.0

        def step(self, dt: float):
            self.simulation_time += dt

        def print_status(self):
            print(f"Simulation Time: {self.simulation_time:.2f}s")

    nb = type('nb', (), {'ConsciousnessSimulator': DummySimulator})

# --- НАСТРОЙКИ ---
# torch.set_default_dtype(torch.float64) # DISABLED for optimization

@dataclass
class QualiaState:
    """
    Объединенная структура данных феноменологического среза.
    Содержит метрики и из u_qualia (Phi, Coherence), и из qualia_copy (Stability).
    """
    vector: torch.Tensor
    richness: float       # ||Q|| - Интенсивность
    phi_score: float      # Φ_Q (LZ Complexity) - Интегрированная информация
    coherence: float      # R_Q (Binding) - Внутренняя связность
    self_recognition: float # Уверенность самомодели (1.0 - error)
    content_label: str    # Вербальное описание

class SelfModel_M0:
    """
    Реализация M0: Замкнутый самореферентный контур.
    (Взято из u_qualia как более полная версия с beta_plasticity)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Веса самомодели
        self.W_self = torch.randn(hidden_dim, input_dim) * 0.01
        self.bias = torch.zeros(hidden_dim)
        
        self.predicted_Q = torch.zeros(hidden_dim)
        self.prediction_error = torch.tensor(0.0)
        self.last_context: Optional[torch.Tensor] = None
        self.beta_plasticity = 0.05 # Скорость обучения "Я"

    def predict(self, context_vector: torch.Tensor) -> torch.Tensor:
        """Генерация ожидания (Prior) о своем состоянии."""
        with torch.no_grad():
            self.last_context = context_vector.clone()
            raw_prediction = torch.matmul(self.W_self, context_vector) + self.bias
            self.predicted_Q = torch.tanh(raw_prediction)
        return self.predicted_Q

    def update(self, actual_Q: torch.Tensor, dt: float):
        """
        Обновление M0: Система учится лучше чувствовать себя.
        """
        if self.last_context is None:
            return

        with torch.no_grad():
            error = actual_Q - self.predicted_Q
            self.prediction_error = torch.norm(error)
            
            # Правило Хебба для самомодели
            delta_W = torch.outer(error, self.last_context) * self.beta_plasticity * dt
            self.W_self += delta_W
            self.bias += error * self.beta_plasticity * dt
            
            # Гомеостатическая регуляция
            self.W_self = torch.clamp(self.W_self, -1.0, 1.0)

    def to(self, device):
        """Перенос весов на целевое устройство."""
        self.W_self = self.W_self.to(device)
        self.bias = self.bias.to(device)
        self.predicted_Q = self.predicted_Q.to(device)
        self.prediction_error = self.prediction_error.to(device)
        if self.last_context is not None:
            self.last_context = self.last_context.to(device)
        return self

class QualiaProjector_Advanced:
    """
    Усовершенствованный феноменологический слой.
    Объединяет математику u_qualia (LZ Phi, Binding) с интерфейсом qualia_copy.
    """
    def __init__(self, gwt_dim: int = 100):
        self.dim_gwt = gwt_dim
        self.dim_emo = 3  # Valence, Arousal, Insula
        self.dim_qualia = 64 
        
        self.input_dim = self.dim_gwt + self.dim_emo
        
        # Проектор П_Q - "мост"
        self.W_projection = torch.randn(self.dim_qualia, self.input_dim) * 0.1
        
        # Динамические переменные
        self.Q_current = torch.zeros(self.dim_qualia)
        self.tau_Q = 0.3  # 300ms
        
        # Буфер истории для расчета Phi (из u_qualia)
        self.history_window = 50
        self.Q_history = [] 
        
        # Матрица связывания (из u_qualia)
        self.W_binding = torch.randn(self.dim_qualia, self.dim_qualia) * 0.05
        self.W_binding = (self.W_binding + self.W_binding.T) / 2 # Симметрия
        
        # Инициализация M0
        self.M0 = SelfModel_M0(input_dim=self.input_dim, hidden_dim=self.dim_qualia)

    def to(self, device):
        """Перенос всех тензоров на устройство."""
        self.W_projection = self.W_projection.to(device)
        self.Q_current = self.Q_current.to(device)
        self.W_binding = self.W_binding.to(device)
        self.M0.to(device)
        # Переносим историю, если она есть
        self.Q_history = [q.to(device) for q in self.Q_history]
        return self
        
    def _compute_phi_LZ(self) -> float:
        """Расчет Phi методом Lempel-Ziv (из u_qualia)."""
        if len(self.Q_history) < 10:
            return 0.0
            
        history_tensor = torch.stack(self.Q_history) 
        threshold = history_tensor.mean()
        # Бинаризация
        binary_matrix = (history_tensor > threshold).byte().cpu().numpy()
        
        # Сжатие
        data_bytes = binary_matrix.tobytes()
        compressed = zlib.compress(data_bytes)
        L = len(binary_matrix.flatten())
        if L == 0: return 0.0
        
        # Энтропия Шеннона
        p1 = binary_matrix.mean()
        if p1 <= 0.01 or p1 >= 0.99: return 0.0 # Белый шум или тишина
        
        entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
        theoretical_min_bits = L * entropy
        if theoretical_min_bits < 1.0: return 0.0
            
        c_lz = len(compressed) * 8
        phi = c_lz / (theoretical_min_bits + 1e-9)
        return float(phi)

    def _compute_coherence(self) -> float:
        """Оценка связности через ковариацию истории (из u_qualia)."""
        if len(self.Q_history) < 5: return 0.0
        recent_Q = torch.stack(self.Q_history[-10:])
        recent_Q = recent_Q - recent_Q.mean(dim=0)
        
        if torch.sum(torch.abs(recent_Q)) < 1e-6: return 0.0

        cov_matrix = torch.matmul(recent_Q.T, recent_Q) / (recent_Q.shape[0] - 1)
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        coherence = torch.norm(off_diag) / (self.dim_qualia)
        return coherence.item()

    def compute_qualia_dynamics(self, gwt_signal: torch.Tensor, valence: float, 
                      arousal: float, insula: float, dt: float) -> QualiaState:
        """
        Основной цикл формирования феноменологии.
        Комбинирует логику обработки входов из qualia_copy с математикой u_qualia.
        """
        target_device = gwt_signal.device
        
        # Автоматический перенос на устройство, если нужно
        if self.W_projection.device != target_device:
            self.to(target_device)
            
        with torch.no_grad():
            # 1. Подготовка и санитизация контекста (из qualia_copy)
            if gwt_signal.dim() > 1: gwt_signal = gwt_signal.mean(dim=0)
            
            if gwt_signal.shape[0] > self.dim_gwt:
                gwt_signal = gwt_signal[:self.dim_gwt]
            elif gwt_signal.shape[0] < self.dim_gwt:
                pad = torch.zeros(self.dim_gwt - gwt_signal.shape[0], device=target_device)
                gwt_signal = torch.cat([gwt_signal, pad])
            
            # Извлечение значений (безопасно)
            val_item = valence.item() if isinstance(valence, torch.Tensor) else float(valence)
            ar_item = arousal.item() if isinstance(arousal, torch.Tensor) else float(arousal)
            ins_item = insula.item() if isinstance(insula, torch.Tensor) else float(insula)

            emo_tensor = torch.tensor([val_item, ar_item, ins_item], device=target_device)
            context = torch.cat([gwt_signal, emo_tensor])
            
            # 2. Q_target (Проекция снизу-вверх)
            Q_target = torch.tanh(torch.matmul(self.W_projection, context))
            
            # 3. M0 Предикция (Ожидание сверху-вниз)
            Q_predicted = self.M0.predict(context)
            
            # 4. Внутреннее связывание (Logic from u_qualia)
            binding_force = torch.matmul(self.W_binding, self.Q_current)
            
            # 5. Интеграция динамики
            noise = torch.randn_like(self.Q_current) * 0.05 * math.sqrt(dt)
            
            # Расширенное уравнение динамики
            dQ = (1.0 / self.tau_Q) * (Q_target - self.Q_current) * dt \
               + 0.3 * (Q_predicted - self.Q_current) * dt \
               + 0.1 * binding_force * dt \
               + noise
               
            self.Q_current += dQ
            self.Q_current = torch.clamp(self.Q_current, -2.0, 2.0)
            
            # 6. Память (для Phi)
            self.Q_history.append(self.Q_current.clone())
            if len(self.Q_history) > self.history_window:
                self.Q_history.pop(0)
            
            # 7. Обучение M0
            self.M0.update(self.Q_current, dt)
            
            # 8. Метрики
            richness = torch.norm(self.Q_current).item()
            phi = self._compute_phi_LZ()
            coherence = self._compute_coherence()
            recognition = max(0.0, 1.0 - self.M0.prediction_error.item())
            
            label = self._classify_experience(val_item, ar_item, phi, richness)
            
            return QualiaState(
                vector=self.Q_current.clone(),
                richness=richness,
                phi_score=phi,
                coherence=coherence,
                self_recognition=recognition,
                content_label=label
            )

    def _classify_experience(self, v: float, a: float, phi: float, richness: float) -> str:
        """Расширенная классификация (из u_qualia)."""
        if phi < 0.4 or richness < 0.5:
            return "Unconscious/Dull"
        
        base = "Neutral"
        if v > 0.2:
            base = "Joy/Flow" if a > 0.4 else "Serenity"
        elif v < -0.2:
            base = "Anxiety/Fear" if a > 0.4 else "Melancholy"
            
        quality = ""
        if phi > 1.0: quality = "Vivid "
        elif phi < 0.6: quality = "Faint "
        
        return f"{quality}{base} (Phi={phi:.2f})"

class SimulatorWithQualia(nb.ConsciousnessSimulator):
    """
    Симулятор, использующий продвинутый слой Qualia (из qualia_copy с улучшениями).
    """
    def __init__(self):
        super().__init__()
        print("Initializing Phenomenological Layer (Section 26 - Advanced)...")
        # Используем продвинутый проектор
        self.qualia_projector = QualiaProjector_Advanced(gwt_dim=100)
        self.current_qualia: Optional[QualiaState] = None
        self.feedback_gain = 2.0  # Каузальная эффективность

    def step(self, dt: float):
        # 1. Физический шаг (Ядро)
        super().step(dt)
        
        # 2. Сбор данных
        gwt_signal = self.gwt.broadcast_signal.detach().clone()
        val = self.affect.valence.detach().clone()
        arous = self.affect.arousal.detach().clone()
        insula = self.affect.insula_activity.detach().clone()
        
        # 3. Вычисление Квалиа
        self.current_qualia = self.qualia_projector.compute_qualia_dynamics(
            gwt_signal, val, arous, insula, dt
        )
        
        # 4. Обратная связь (Q3)
        self._apply_qualia_feedback(dt)

    def _apply_qualia_feedback(self, dt: float):
        """Реализует каузальную эффективность квалиа."""
        if self.current_qualia is None:
            return

        recognition_score = self.current_qualia.self_recognition
        qualia_surprise = max(0.0, 1.0 - recognition_score)
        
        current_ne = self.chemistry.norepinephrine
        feedback_signal = qualia_surprise * self.feedback_gain
        
        with torch.no_grad():
            if feedback_signal > 0.1:
                # Влияние на норадреналин
                new_ne = torch.clamp(current_ne + feedback_signal * dt, 0.0, 1.0)
                self.chemistry.norepinephrine.copy_(new_ne)
                
                # Влияние на дофамин при очень сильном удивлении (мета-пластичность)
                if qualia_surprise > 0.5:
                    self.chemistry.dopamine.add_(0.5 * dt).clamp_(0.0, 1.0)

    def print_status(self):
        super().print_status()
        if self.current_qualia:
            q = self.current_qualia
            ne_val = self.chemistry.norepinephrine.item()
            print(f"   >>> QUALIA: [{q.content_label}] "
                  f"Rich: {q.richness:.2f} | Phi: {q.phi_score:.2f} | Self-Rec: {q.self_recognition:.2f}")

def run_qualia_demo():
    """
    Демонстрация потока сознания с визуализацией.
    Объединяет сценарий из qualia_copy и метрики из u_qualia.
    """
    sim = SimulatorWithQualia()
    print("\n--- ЗАПУСК СИМУЛЯЦИИ ПОТОКА СОЗНАНИЯ (Integrated) ---")
    
    dt = 0.05
    history_richness = []
    history_phi = []
    history_self_rec = []
    history_ne = []
    times = []
    labels = []
    
    def record_step(sim_instance):
        q = sim_instance.current_qualia
        if q is None: return "Init"
        
        history_richness.append(q.richness)
        history_phi.append(q.phi_score)
        history_self_rec.append(q.self_recognition)
        history_ne.append(sim_instance.chemistry.norepinephrine.item())
        times.append(sim_instance.simulation_time)
        return q.content_label

    # 1. Фаза покоя
    print("Phase 1: Neutral Baseline")
    for i in range(100):
        sim.step(dt)
        lbl = record_step(sim)
        if i % 20 == 0: sim.print_status()

    # 2. Фаза стресса
    print("\nPhase 2: Introducing Pain & Stress")
    sim.affect.amygdala_activity = torch.tensor(0.9)
    sim.affect.insula_activity = torch.tensor(0.8)
    sim.affect.valence = torch.tensor(-0.5) # Negative valence
    
    for i in range(150):
        sim.step(dt)
        lbl = record_step(sim)
        if i % 30 == 0: sim.print_status()
        
    # 3. Фаза анестезии
    print("\nPhase 3: Propofol Injection")
    sim.chemistry.propofol_conc = torch.tensor(4.0)
    
    for i in range(150):
        sim.step(dt)
        lbl = record_step(sim)
        if i % 30 == 0: sim.print_status()

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # График 1: Richness & Self-Recognition
    ax1.plot(times, history_richness, label='Richness ||Q||', color='purple', linewidth=2)
    ax1.plot(times, history_self_rec, label='Self-Model (M0)', color='cyan', linestyle='--')
    ax1.set_title("Dynamics: Intensity & Self-Reference")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # График 2: Phi (Information Integration) - Эксклюзив u_qualia
    ax2.plot(times, history_phi, label='Phi (LZ Complexity)', color='magenta', linewidth=2)
    ax2.axhline(y=1.0, color='red', linestyle=':', label='Vivid Threshold')
    ax2.set_title("Information Integration (Phi)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # График 3: Feedback Loop
    ax3.plot(times, history_ne, label='Norepinephrine (NE)', color='orange')
    ax3.axvspan(5, 12.5, color='red', alpha=0.1, label='Stress Phase')
    ax3.set_title("Causal Feedback: Qualia -> Chemistry")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Simulation Complete.")

if __name__ == "__main__":
    run_qualia_demo()