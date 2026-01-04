import torch
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# Импортируем базовые классы из вашего оригинального файла
# Предполагается, что ваш файл называется A_NeuralBiocore_vDeltha.py
# Если нет, измените импорт ниже.
from Neuralbiocore_U_for_GPU import (
    BioChemistry, AffectiveSystem, PhysicsConfig, ChemistryConfig,
    BrainHierarchy, GlobalWorkspace, BodyAgent, SleepCycleManager,
    NeuralLayer # Needed for typing
)

# Устанавливаем точность
# torch.set_default_dtype(torch.float64) # DISABLED for optimization

@dataclass
class SocialModel:
    """Структура для хранения ментальной модели агента (M_x)"""
    intentions: torch.Tensor  # Вектор намерений (например, кооперация vs дефекция)
    beliefs: torch.Tensor     # Убеждения о мире
    emotional_state: torch.Tensor # [Valence, Arousal]
    confidence: float         # Уверенность в этой модели (Precision)

class AdvancedSocialCognition:
    """
    Реализация раздела 19: Theory of Mind и Рекурсия.
    Поддерживает иерархию M0 (Я), M1 (Другой), M2 (Я глазами Другого).
    """
    def __init__(self, vector_dim: int = 10):
        self.dim = vector_dim
        
        # M0: Self-Model (Прямой доступ к собственным намерениям)
        self.M0 = SocialModel(
            intentions=torch.zeros(self.dim),
            beliefs=torch.zeros(self.dim),
            emotional_state=torch.zeros(2),
            confidence=1.0
        )
        
        # M1: Model of Other (Инференция скрытых состояний другого)
        # "Что я думаю о нем"
        self.M1_other = SocialModel(
            intentions=torch.zeros(self.dim), # Начальное предположение (neutral)
            beliefs=torch.zeros(self.dim),
            emotional_state=torch.zeros(2),
            confidence=0.5 # Низкая уверенность в начале
        )
        
        # M2: Meta-Model / Recursion (Self-in-Other)
        # "Что я думаю, что он думает обо мне"
        self.M2_meta = SocialModel(
            intentions=torch.zeros(self.dim),
            beliefs=torch.zeros(self.dim),
            emotional_state=torch.zeros(2),
            confidence=0.5
        )
        
        # Параметры обучения (Inverse Inference)
        self.learning_rate_ToM = 0.15
        self.prediction_error_history: List[float] = []

    def update_M0_self(self, my_action_plan: torch.Tensor, val: float, ar: float):
        """Обновление модели себя (Ground Truth)"""
        with torch.no_grad():
            # Скользящее среднее намерений (интеграция во времени)
            if my_action_plan.shape[0] >= self.dim:
                target = my_action_plan[:self.dim]
            else:
                target = torch.cat([my_action_plan, torch.zeros(self.dim - my_action_plan.shape[0], device=my_action_plan.device)])
            
            if self.M0.intentions.device != my_action_plan.device:
                self.M0.intentions = self.M0.intentions.to(my_action_plan.device)
                self.M0.beliefs = self.M0.beliefs.to(my_action_plan.device)
                self.M0.emotional_state = self.M0.emotional_state.to(my_action_plan.device)
                
            self.M0.intentions += 0.2 * (target - self.M0.intentions)
            self.M0.emotional_state = torch.tensor([val, ar], device=my_action_plan.device)

    def infer_M1_other(self, observed_action: torch.Tensor, context_vector: Optional[torch.Tensor] = None):
        """
        Inverse Inference (Обратный вывод):
        Наблюдая действие O(t), обновляем M1(t).
        Раздел 19b: p(M_other | O) propto p(O | M_other) * p(M_other)
        """
        with torch.no_grad():
            # 1. Приведение размерности
            if observed_action.shape[0] >= self.dim:
                obs = observed_action[:self.dim]
            else:
                obs = torch.cat([observed_action, torch.zeros(self.dim - observed_action.shape[0], device=observed_action.device)])
            
            if self.M1_other.intentions.device != observed_action.device:
                self.M1_other.intentions = self.M1_other.intentions.to(observed_action.device)
                self.M1_other.beliefs = self.M1_other.beliefs.to(observed_action.device)
                self.M1_other.emotional_state = self.M1_other.emotional_state.to(observed_action.device)

            # 2. Предиктивное кодирование в социальном слое
            # Прогноз действия на основе текущей модели M1
            predicted_action = self.M1_other.intentions
            
            # 3. Ошибка предсказания (Social Prediction Error - SPE)
            spe = obs - predicted_action
            spe_magnitude = torch.norm(spe).item()
            self.prediction_error_history.append(spe_magnitude)
            
            # 4. Обновление убеждений (Update Beliefs via Error Minimization)
            # Если ошибка большая -> снижаем уверенность, меняем модель быстрее
            adaptive_lr = self.learning_rate_ToM * (1.0 + spe_magnitude)
            
            self.M1_other.intentions += adaptive_lr * spe
            
            # 5. Вывод эмоций другого по "энергии" действия (эвристика)
            # В полной версии здесь должен быть анализ лицевой экспрессии
            action_energy = torch.norm(obs)
            inferred_arousal = torch.tanh(action_energy)
            # Если действие резкое и непредсказуемое -> возможно гнев/страх (Valence down)
            inferred_valence = 0.5 - spe_magnitude 
            
            self.M1_other.emotional_state = torch.tensor([inferred_valence, inferred_arousal], device=observed_action.device)

    def update_M2_recursion(self, my_last_action: torch.Tensor):
        """
        Обновление M2: Как мои действия влияют на мнение другого обо мне?
        "Если я ударил его (action), он теперь думает, что я враг (M2 update)"
        """
        with torch.no_grad():
            # Упрощенная логика зеркалирования:
            # Я предполагаю, что он использует такой же механизм вывода, как и я.
            # M2(t+1) ~ M2(t) + lr * (My_Action - M2_expectations)
            
            if self.M2_meta.intentions.device != my_last_action.device:
                self.M2_meta.intentions = self.M2_meta.intentions.to(my_last_action.device)
                self.M2_meta.beliefs = self.M2_meta.beliefs.to(my_last_action.device)
                self.M2_meta.emotional_state = self.M2_meta.emotional_state.to(my_last_action.device)

            if my_last_action.shape[0] >= self.dim:
                act = my_last_action[:self.dim]
            else:
                act = torch.cat([my_last_action, torch.zeros(self.dim - my_last_action.shape[0], device=my_last_action.device)])
            
            # Если я действую агрессивно, я обновляю M2, полагая, что он видит меня агрессивным
            self.M2_meta.intentions += 0.1 * (act - self.M2_meta.intentions)

    def calculate_social_pain(self) -> float:
        """
        Социальная боль (Раздел 21e):
        Расхождение между M2 (как меня видят) и Идеальным Я (или потребностью в принадлежности).
        """
        # Допустим, мы хотим, чтобы нас видели кооперирующими (intentions > 0)
        # Если M2 показывает, что нас видят как врага (intentions < 0) -> Боль
        
        perceived_reputation = torch.mean(self.M2_meta.intentions).item()
        
        # Если репутация негативная, возникает боль
        rejection_signal = max(0.0, -perceived_reputation)
        
        # Также боль от непонимания (высокая SPE)
        uncertainty_pain = 0.0
        if len(self.prediction_error_history) > 5:
            uncertainty_pain = np.mean(self.prediction_error_history[-5:]) * 0.2
            
        return rejection_signal + uncertainty_pain
    
    def get_social_pain_signal(self) -> float:
        """Алиас для совместимости с GlobalWorkspace"""
        return self.calculate_social_pain()

    @property
    def m2_self_in_other(self) -> torch.Tensor:
        """Алиас для доступа к вектору намерений мета-модели"""
        return self.M2_meta.intentions

class SimulatorWithToM(GlobalWorkspace): 
    # Это вспомогательный класс-обертка или расширение Simulator
    # В данном контексте мы просто расширим основной класс ConsciousnessSimulator
    pass

class AgentSimulator:
    """
    Обертка над ConsciousnessSimulator для агентов.
    """
    def __init__(self, name: str):
        self.name = name
        # Инициализация ядра
        self.phys_cfg = PhysicsConfig()
        self.chem_cfg = ChemistryConfig()
        self.chemistry = BioChemistry(self.chem_cfg)
        self.sleep_manager = SleepCycleManager(self.chemistry)
        
        # Важно: BodyAgent теперь инициализируется с 1 актуатором по умолчанию, 
        # но мы можем расширить это в будущем.
        self.body = BodyAgent(n_sensors=1000, n_actuators=1)
        
        self.hierarchy = BrainHierarchy(self.phys_cfg, self.chemistry, input_dim=1000)
        self.gwt = GlobalWorkspace(self.hierarchy)
        self.affect = AffectiveSystem(self.chemistry)
        
        # ЗАМЕНА: Используем продвинутый ToM
        self.social = AdvancedSocialCognition(vector_dim=10)
        
        self.simulation_time = 0.0
        self.environment_target = 0.5

    def step_internal(self, dt: float, sensory_input_override: Optional[torch.Tensor] = None):
        """Один шаг внутренней физики мозга"""
        self.simulation_time += dt
        
        # 1. Сенсорика
        if sensory_input_override is not None:
            self.body.sensory_input = sensory_input_override
        else:
            # Стандартный шум, если нет входа
            self.body.sensory_input *= 0.9

        # Получаем текущее действие (ВЕКТОР)
        current_action_vec = self.body.get_action_vector()
            
        # 2. Иерархия (Predictive Coding)
        # В updated core передаем action_vector в иерархию
        global_feedback = self.gwt.get_context_feedback()
        self.hierarchy.process_sensory_input(
            sensory_input=self.body.sensory_input, 
            action_vector=current_action_vec, # <-- ВАЖНО: передаем вектор действия
            dt=dt, 
            sleep_stage="Wake", 
            global_context=global_feedback
        )
        
        # 3. GWT & Мысли
        # В GWT подаем сигналы от Социального модуля (мысли о другом)
        self.gwt.step(dt, social_module=None) 
        
        # 4. Эмоции и Химия
        social_pain = self.social.calculate_social_pain()
        
        # Интеграция: ToM влияет на нейрохимию (раздел 20d)
        if social_pain > 0.1:
            self.chemistry.norepinephrine = torch.clamp(self.chemistry.norepinephrine + dt * 0.5, 0, 1)
            self.affect.insula_activity = torch.tensor(0.8)
        
        total_fe = self.hierarchy.get_global_free_energy()
        self.affect.update(dt, total_fe, social_pain, reward_signal=0.0)
        self.chemistry.update(dt, self.affect.arousal, 0.0)
        
        # 5. Обновление M0 (Self)
        # Преобразуем вектор действия (размер 1) в вектор социальной модели (размер 10)
        if current_action_vec.numel() == 1:
            my_action_projection = current_action_vec.expand(10)
        else:
            # Если актуаторов больше, берем среднее или паддинг (упрощение)
            my_action_projection = torch.cat([current_action_vec, torch.zeros(10 - current_action_vec.numel(), device=current_action_vec.device)])

        self.social.update_M0_self(my_action_projection, self.affect.valence.item(), self.affect.arousal.item())

    def perceive_social_signal(self, other_action: float, dt: float):
        """Восприятие действия другого агента"""
        obs_vector = torch.ones(10) * other_action
        self.social.infer_M1_other(obs_vector)
        
    def decide_action(self) -> float:
        """
        В этой архитектуре 'decide_action' скорее является наблюдателем за решениями Active Inference,
        либо корректирующим сигналом высокого уровня.
        """
        base_intent = torch.mean(self.social.M0.intentions).item()
        other_intent = torch.mean(self.social.M1_other.intentions).item()
        ne_level = self.chemistry.norepinephrine.item()
        
        # a = w1 * Trust - w2 * Fear
        decision = 0.8 * other_intent - 0.5 * (ne_level - 0.5)
        decision += np.random.normal(0, 0.1)
        return max(-1.0, min(1.0, decision))
    
    def apply_reward(self, reward: float, dt: float):
        """Получение награды (Dopamine update)"""
        current_val = self.affect.valence.item()
        rpe = reward - current_val 
        
        if rpe > 0:
            self.chemistry.dopamine = torch.clamp(self.chemistry.dopamine + rpe * dt, 0, 1)
            self.affect.valence = torch.clamp(self.affect.valence + dt, -1, 1)
        else:
            self.chemistry.dopamine = torch.clamp(self.chemistry.dopamine - 0.1 * dt, 0, 1)
            self.affect.valence = torch.clamp(self.affect.valence - dt, -1, 1)

