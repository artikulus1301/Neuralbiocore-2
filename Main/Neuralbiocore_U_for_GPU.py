import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

if torch.cuda.is_available():
    print("GPU найден!")
    print("Имя:", torch.cuda.get_device_name(0))
    print("CUDA версия:", torch.version.cuda)
    print("Количество GPU:", torch.cuda.device_count())
else:
    print("GPU не найден, используется CPU")

import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm
import zlib
import json

"""
NeuralBiocore_U.py 
Не просто in-silico имплементация Математической Модели Сознания но и будущий фундамент для нейроморфного искусственного интеллекта.
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

# torch.set_default_dtype(torch.float64) # REMOVED: Optimize memory
torch.set_float32_matmul_precision('high') # Enable TF32 for Speed

# ==========================================
# 0. JIT KERNELS (PyTorch 2.0+)
# ==========================================

#torch.compile(mode="reduce-overhead")
def dynamics_kernel(V, phase, spikes, refractory_timer, ATP, is_dead,
                   I_drive_total, 
                   s_gaba_a, s_gaba_b,
                   gaba_concentration,
                   natural_omegas,
                   chem_dopamine, 
                   dt, tau_mem, v_rest, v_threshold, refractory_period,
                   c_spike, c_recovery, c_baseline, critical_atp,
                   tau_gaba_a, tau_gaba_b, g_gaba_a_mod,
                   coupling_strength, alpha_sync, general_suppression,
                   tonic_inhibition):
    
    # 1. Energy Dynamics
    live_mask = ~is_dead
    can_spike = (ATP > c_spike) & live_mask
    
    # In-place updates for ATP
    # ATP.add_(c_recovery - c_baseline) # Global update (masked later?)
    # Logic: ATP = ATP + c_recovery - c_baseline if live
    
    # Calculate delta for live neurons
    delta_atp = torch.zeros_like(ATP)
    delta_atp.masked_fill_(live_mask, c_recovery - c_baseline)
    ATP.add_(delta_atp)
    
    # Subtract cost of spikes
    # ATP = ATP - spikes.float() * c_spike
    ATP.sub_(spikes.float() * c_spike)
    
    dying_now = (ATP < critical_atp) & live_mask
    # Apply death effects
    is_dead.logical_or_(dying_now)
    
    # V = torch.where(dying_now, 0.2, V)
    V.masked_fill_(dying_now, 0.2)
    
    # spikes = torch.where(dying_now, False, spikes)
    spikes.masked_fill_(dying_now, False)
    
    # phase = torch.where(dying_now, 0.0, phase)
    phase.masked_fill_(dying_now, 0.0)
    
    live_mask = ~is_dead
    
    ATP.clamp_(0.0, 1.0)
    
    # 2. GABA Dynamics
    decay_a = (1.0 - dt / tau_gaba_a)
    growth_a = dt * 10.0 * gaba_concentration
    s_gaba_a.mul_(decay_a).add_(growth_a * (1.0 - s_gaba_a)).clamp_(0.0, 1.0)
    
    decay_b = (1.0 - dt / tau_gaba_b)
    growth_b = dt * 5.0 * gaba_concentration
    s_gaba_b.mul_(decay_b).add_(growth_b * (1.0 - s_gaba_b)).clamp_(0.0, 1.0)
    
    # 3. Neuron Dynamics (LIF) - In-place accumulation
    dv = I_drive_total - tonic_inhibition
    dv.sub_(V)
    
    # GABA Effect
    total_conductance = (s_gaba_a * g_gaba_a_mod).add_(s_gaba_b * 0.5)
    dv.sub_(total_conductance * (V + 0.7))
    dv.mul_(dt / tau_mem)
    
    not_refractory = (refractory_timer <= 0)
    update_mask = not_refractory & live_mask
    
    # V += dv where update_mask
    V.masked_scatter_(update_mask, V[update_mask] + dv[update_mask])
    V.clamp_(-2.0, 5.0)
    
    # refractory_timer -= dt
    refractory_timer.sub_(dt).clamp_(min=0.0)
    
    # 4. Phase Dynamics
    complex_phases = torch.exp(1j * phase)
    z = torch.mean(complex_phases)
    R = torch.abs(z)
    Psi = torch.angle(z)
    
    effective_coupling_val = coupling_strength * general_suppression
    freq_mod = 1.0 / (1.0 + 0.3 * (s_gaba_a + s_gaba_b))
    omega_eff = natural_omegas * freq_mod * (1 + 0.15 * chem_dopamine)
    
    # dphi
    dphi = torch.sin(Psi - phase).mul_(effective_coupling_val * R)
    dphi.add_(omega_eff).add_(V * 0.02).mul_(dt)
    
    # Update phase only for live neurons
    phase_update = torch.remainder(phase + dphi, 2 * math.pi)
    phase.masked_scatter_(live_mask, phase_update[live_mask])

    # 5. Spiking
    base_threshold = v_threshold
    adaptive_threshold = base_threshold * (1.0 + 0.2 * s_gaba_a)
    phase_mod = 1.0 + alpha_sync * torch.cos(phase)
    effective_threshold = adaptive_threshold / phase_mod
    
    new_spikes = (V > effective_threshold) & not_refractory & can_spike
    
    # Reset
    V.masked_fill_(new_spikes, v_rest)
    refractory_timer.masked_fill_(new_spikes, refractory_period)
    
    return V, phase, spikes, new_spikes, refractory_timer, ATP, is_dead, s_gaba_a, s_gaba_b

# @torch.compile(fullgraph=False) # DISABLED: Requires MSVC/C++ compiler on Windows
def fused_layer_step(
    V, phase, spikes, refractory_timer, ATP, is_dead,
    s_gaba_a, s_gaba_b,
    V_PV, V_SST, V_VIP, 
    I_exc, I_ext, I_mirror,
    chem_dopamine, chem_acetylcholine, propofol_conc,
    global_context,
    dt, 
    tau_mem, v_rest, v_threshold, refractory_period,
    c_spike, c_recovery, c_baseline, critical_atp,
    tau_gaba_a, tau_gaba_b, g_gaba_a_mod,
    coupling_strength, alpha_sync,
    natural_omegas,
    initial_atp 
):
    # 1. Chemistry & Modulation
    p = propofol_conc
    general_suppression = torch.exp(-p / 1.9)
    paradoxical_boost = 0.8 * torch.exp(-((p - 2.5) ** 2) / 0.5)
    drive_modulator = torch.clamp(general_suppression + paradoxical_boost, 0.01, 2.3)
    interneuron_suppression = 1.0 / (1.0 + 1.5 * paradoxical_boost)

    mean_spikes = spikes.float().mean()

    # 2. Interneurons (Inlined)
    # PV
    V_PV.add_(dt * (-V_PV + mean_spikes * 15.0 * interneuron_suppression))
    spikes_pv = V_PV > 1.0
    V_PV.masked_fill_(spikes_pv, 0.0)
    
    # SST
    inhibition_from_vip = (V_VIP > 0.8).float().mean() * 5.0
    V_SST.add_(dt * (-V_SST + mean_spikes * 8.0 * interneuron_suppression - inhibition_from_vip))
    spikes_sst = V_SST > 1.0
    V_SST.masked_fill_(spikes_sst, 0.0)
    
    # VIP
    V_VIP.add_(dt * (-V_VIP + 0.15) + torch.randn_like(V_VIP) * 0.1 * math.sqrt(dt))

    # GABA Concentration
    pv_act = spikes_pv.float().mean()
    sst_act = spikes_sst.float().mean()
    raw_conc = (pv_act * 1.2 + sst_act) * 8.0 
    gaba_concentration = torch.clamp(raw_conc, max=6.0)

    # 3. Input Accumulation (Fused)
    I_total = I_exc + I_ext + I_mirror 
    
    # Background
    I_background = 2.0 * chem_acetylcholine + 0.5
    I_total.add_(I_background)
    
    # Global Context
    if global_context is not None:
        if global_context.numel() > 1:
             if global_context.shape[0] == V.shape[0]:
                I_total.add_(global_context * 2.0)
             else:
                I_total.add_(global_context.mean() * 2.0)
        else:
             I_total.add_(global_context * 2.0)
             
    # Noise
    sqrt_dt = math.sqrt(dt)
    noise = torch.randn_like(V) * (1.2 * drive_modulator * sqrt_dt)
    I_total.add_(noise)
    
    # Modulation
    I_total.mul_(drive_modulator)

    # 4. Call Kernel
    (
        V, phase, spikes, new_spikes, refractory_timer, ATP, is_dead, s_gaba_a, s_gaba_b
    ) = dynamics_kernel(
        V, phase, spikes, refractory_timer, ATP, is_dead,
        I_total, 
        s_gaba_a, s_gaba_b,
        gaba_concentration,
        natural_omegas,
        chem_dopamine, 
        dt, 
        tau_mem, v_rest, v_threshold, refractory_period,
        c_spike, c_recovery, c_baseline, critical_atp,
        tau_gaba_a, tau_gaba_b, g_gaba_a_mod,
        coupling_strength, alpha_sync, general_suppression,
        propofol_conc * 2.0 
    )
    
    return V, phase, spikes, new_spikes, refractory_timer, ATP, is_dead, s_gaba_a, s_gaba_b, V_PV, V_SST, V_VIP

# ==========================================
# 1. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ==========================================

@dataclass
class TopologyConfig:
    """Small-World network topology configuration"""
    topology_type: str = "small_world"
    k_local: int = 20
    p_rewire: float = 0.08
    shortcut_density: float = 0.03
    bottom_up_locality: int = 15
    top_down_locality: int = 35
    local_boost: float = 2.031
    enable_modules: bool = True
    n_modules: int = 4
    inter_module_sparsity: float = 0.85

@dataclass
class PhysicsConfig:
    """Параметры физики и времени"""
    dt: float = 0.001
    N_neurons: int = 1000
    
    # Мембранные параметры
    tau_mem: float = 0.05
    v_rest: float = 0.0
    v_threshold: float = 1.1
    refractory_period: float = 0.005
    
    # Фазовая динамика (Гамма-ритм ~40Гц)
    omega_base: float = 2 * math.pi * 40.0
    omega_std: float = 2 * math.pi * 0.5
    coupling_strength: float = 0.5
    alpha_sync: float = 0.3
    
    # Энергия и Смерть
    initial_atp: float = 1.0
    c_spike: float = 0.001
    c_synapse: float = 0.001
    c_baseline: float = 0.0001
    c_recovery: float = 0.002
    critical_atp: float = 0.01

@dataclass
class NeurogenesisConfig:
    """
    Конфигурация динамического развития мозга.
    """
    # Пределы (OOM Protection)
    max_neurons_per_layer: int = 10000  # Verified Safe Limit (80% of OOM crash at 12k)
    
    # Фаза 1: Быстрый рост (Morphogenesis)
    initial_neurons: int = 100         # С чего начинаем (эмбрион)
    base_target_neurons: int = 8000    # Цель "созревания"
    growth_rate_fast: int = 50         # Нейронов за шаг обновления
    
    # Фаза 2: Медленный рост (Adult Plasticity)
    growth_rate_slow: int = 5          # Нейронов при адаптации
    error_threshold_trigger: float = 0.5 # Уровень Free Energy для триггера роста (с учетом clipping)
    atp_cost_per_neuron: float = 0.8   # Требуемый уровень энергии слоя для деления
    
    # Частота обновлений (в шагах симуляции)
    update_interval: int = 100         # Не ресайзим тензоры каждый тик (дорого)

@dataclass
class ChemistryConfig:
    """Параметры нейромодуляции и анестезии"""
    lambda_decay: float = 0.01
    
    # GABA параметры
    e_cl: float = -0.070
    e_k: float = -0.090
    tau_gaba_a: float = 0.010
    tau_gaba_b: float = 0.200
    
    # Анестезия
    k_propofol: float = 8.0

# ==========================================
# 2. TOPOLOGY GENERATOR (Small-World)
# ==========================================

class TopologyGenerator:
    """Генератор топологии Small-World с модульной структурой"""
    
    @staticmethod
    def create_small_world_mask(n_pre: int, n_post: int, cfg: TopologyConfig, 
                                 layer_type: str = "bottom_up") -> torch.Tensor:
        """
        Создаёт маску связности для Small-World сети.
        
        Args:
            n_pre: количество пресинаптических нейронов
            n_post: количество постсинаптических нейронов
            cfg: конфигурация топологии
            layer_type: тип слоя ("bottom_up" или "top_down")
        
        Returns:
            mask: бинарная матрица связности [n_post, n_pre]
        """
        mask = torch.zeros(n_post, n_pre)
        k = cfg.bottom_up_locality if layer_type == "bottom_up" else cfg.top_down_locality
        
        # 1. Локальное кольцо (Regular lattice)
        for i in range(n_post):
            center_j = int((i / n_post) * n_pre)
            start = max(0, center_j - k)
            end = min(n_pre, center_j + k + 1)
            mask[i, start:end] = 1.0
        
        # 2. Rewiring (Watts-Strogatz) - ОПТИМИЗИРОВАНО: векторизация
        if cfg.p_rewire > 0:
            existing = torch.nonzero(mask)
            n_rewire = int(len(existing) * cfg.p_rewire)
            if n_rewire > 0:
                samples = existing[torch.randperm(len(existing))[:n_rewire]]
                # Batch удаление старых связей
                mask[samples[:, 0], samples[:, 1]] = 0.0
                # Batch создание новых связей
                new_js = torch.randint(0, n_pre, (n_rewire,))
                mask[samples[:, 0], new_js] = 1.0
        
        # 3. Long-range shortcuts - ОПТИМИЗИРОВАНО: векторизация
        n_shortcuts = int(n_pre * n_post * cfg.shortcut_density)
        if n_shortcuts > 0:
            shortcut_i = torch.randint(0, n_post, (n_shortcuts,))
            shortcut_j = torch.randint(0, n_pre, (n_shortcuts,))
            mask[shortcut_i, shortcut_j] = 1.0
        
        # 4. Модульная структура
        if cfg.enable_modules:
            mask = TopologyGenerator._add_modules(mask, n_pre, n_post, cfg)
        
        return mask
    
    @staticmethod
    def _add_modules(mask: torch.Tensor, n_pre: int, n_post: int, cfg: TopologyConfig):
        """Добавляет модульную структуру (подавление межмодульных связей)"""
        mod_size_pre = n_pre // cfg.n_modules
        mod_size_post = n_post // cfg.n_modules
        
        for i in range(cfg.n_modules):
            for j in range(cfg.n_modules):
                if i == j: continue  # Внутримодульные связи не трогаем
                
                post_s, post_e = i * mod_size_post, (i + 1) * mod_size_post
                pre_s, pre_e = j * mod_size_pre, (j + 1) * mod_size_pre
                
                # Подавляем случайную долю межмодульных связей
                suppress = (torch.rand(post_e - post_s, pre_e - pre_s) > cfg.inter_module_sparsity).float()
                mask[post_s:post_e, pre_s:pre_e] *= suppress
        
        return mask

# ==========================================
# 3. СОСТОЯНИЕ НЕЙРОХИМИИ
# ==========================================

class BioChemistry:
    """
    Управляет глобальными и локальными концентрациями нейромедиаторов.
    """
    def __init__(self, config: ChemistryConfig):
        self.cfg = config
        
        # Глобальные нейромодуляторы
        self.dopamine = torch.tensor(0.5, device=DEVICE)
        self.acetylcholine = torch.tensor(0.5, device=DEVICE)
        self.serotonin = torch.tensor(0.5, device=DEVICE)
        self.norepinephrine = torch.tensor(0.5, device=DEVICE)
        
        # Анестезия
        self.propofol_conc = torch.tensor(0.0, device=DEVICE)
    
    def get_gaba_conductance_modifier(self) -> torch.Tensor:
        """Возвращает множитель проводимости GABA_A в зависимости от анестезии"""
        return 1.0 + self.cfg.k_propofol * self.propofol_conc

    def update(self, dt: float, stress_level: float, reward_prediction_error: float):
        """Обновление уровней нейромедиаторов"""
        # DA реагирует на RPE
        delta_da = reward_prediction_error * 0.2 - (self.dopamine - 0.5) * self.cfg.lambda_decay
        self.dopamine = torch.clamp(self.dopamine + delta_da * dt, 0.0, 1.0)
        
        # NE реагирует на стресс
        if not isinstance(stress_level, torch.Tensor):
            stress_level = torch.tensor(stress_level, device=DEVICE)
            
        delta_ne = stress_level * 0.1 - (self.norepinephrine - 0.5) * self.cfg.lambda_decay
        self.norepinephrine = torch.clamp(self.norepinephrine + delta_ne * dt, 0.0, 1.0)

    def get_state(self) -> dict:
        return {
            'dopamine': self.dopamine,
            'acetylcholine': self.acetylcholine,
            'serotonin': self.serotonin,
            'norepinephrine': self.norepinephrine,
            'propofol_conc': self.propofol_conc
        }

    def load_state(self, state: dict):
        self.dopamine = state.get('dopamine', self.dopamine)
        self.acetylcholine = state.get('acetylcholine', self.acetylcholine)
        self.serotonin = state.get('serotonin', self.serotonin)
        self.norepinephrine = state.get('norepinephrine', self.norepinephrine)
        self.propofol_conc = state.get('propofol_conc', self.propofol_conc)

# ==========================================
# 4. НЕЙРОННЫЙ СЛОЙ (ВЕКТОРИЗОВАННЫЙ PyTorch)
# ==========================================

class NeuralLayer:
    """
    Представляет слой нейронов с биофизической динамикой.
    """
    def __init__(self, layer_id: str, n_neurons: int, phys_cfg: PhysicsConfig, chem_sys: BioChemistry):
        self.id = layer_id
        self.N = n_neurons
        self.p_cfg = phys_cfg
        self.chem = chem_sys
        
        # --- 1. Инициализация состояния (CRITICAL: Float64) ---
        self.V = (torch.rand(self.N, device=DEVICE) * 0.5 - 0.2).to(dtype=torch.float64)
        self.phase = (torch.rand(self.N, device=DEVICE) * 2 * math.pi).to(dtype=torch.float64)
        
        # --- НАТУРАЛЬНЫЕ ЧАСТОТЫ ---
        base_freqs = torch.normal(
            self.p_cfg.omega_base, 
            self.p_cfg.omega_std,
            size=(self.N,),
            device=DEVICE
        )
        self.natural_omegas = torch.clamp(
            base_freqs, 
            min=2 * math.pi * 5.0,
            max=2 * math.pi * 100.0
        ).to(dtype=torch.float64)
        
        # --- Инициализация спайков ПЕРЕД первым шагом ---
        self.spikes = torch.zeros(self.N, dtype=torch.bool, device=DEVICE)
        self.refractory_timer = torch.zeros(self.N, device=DEVICE)
        self.smoothed_rate = torch.tensor(0.0, device=DEVICE)
        
        # --- Интернейроны ---
        self.n_interneurons = int(self.N * 0.2)
        self.V_PV = torch.zeros(self.n_interneurons, device=DEVICE)
        self.V_SST = torch.zeros(self.n_interneurons, device=DEVICE)
        self.V_VIP = torch.zeros(self.n_interneurons, device=DEVICE)
        
        # --- Токи ---
        self.I_exc = torch.zeros(self.N, device=DEVICE) 
        self.I_inh = torch.zeros(self.N, device=DEVICE) 
        self.I_ext = torch.zeros(self.N, device=DEVICE) 
        
        # --- Рецепторы ---
        self.s_gaba_a = torch.zeros(self.N, device=DEVICE) 
        self.s_gaba_b = torch.zeros(self.N, device=DEVICE)

        # === ENERGY STATE ===
        self.ATP = (torch.ones(self.N, device=DEVICE) * self.p_cfg.initial_atp).to(dtype=torch.float64)
        self.is_dead = torch.zeros(self.N, dtype=torch.bool, device=DEVICE)
        
        # === MIRROR INPUT ===
        self.I_mirror = torch.zeros(self.N, device=DEVICE)
        
        # === OPTIMIZATION: Предвыделенные буферы для избежания аллокаций каждый шаг ===
        self._noise_buffer = torch.zeros(self.N, device=DEVICE)
        self._I_global_buffer = torch.zeros(self.N, device=DEVICE)
        self._interneuron_noise = torch.zeros(self.n_interneurons, device=DEVICE)
        self._computation_buffer = torch.zeros(self.N, device=DEVICE)

    def compute_kuramoto_order(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление параметра порядка Kuramoto: z = (1/N) * Σ exp(i*θ_j)
        Возвращает (R, Ψ) где R - синхронизация [0,1], Ψ - средняя фаза
        """
        complex_phases = torch.exp(1j * self.phase)
        z = torch.mean(complex_phases)
        
        R = torch.abs(z)
        Psi = torch.angle(z)
        
        return R, Psi

    @torch.no_grad()
    def dynamics_step(self, dt: float, global_context: Optional[torch.Tensor] = None):
        """Обновление состояния слоя - ОПТИМИЗИРОВАНО: JIT Kernel + Zero-Copy Buffers"""
        # DEBUG: Check devices
        # print(f"DEBUG: V device: {self.V.device}")
        # print(f"DEBUG: natural_omegas device: {self.natural_omegas.device}")
        # print(f"DEBUG: dopamine device before cast: {self.chem.dopamine.device}")
        
        # Force natural_omegas to correct device if needed
        if self.natural_omegas.device != self.V.device:
             self.natural_omegas = self.natural_omegas.to(self.V.device)


        # Call the fused kernel
        # We ignore 'prev_spikes' (3rd return) and assign 'new_spikes' (4th return) to self.spikes
        (
            self.V, self.phase, _, self.spikes, 
            self.refractory_timer, self.ATP, self.is_dead, 
            self.s_gaba_a, self.s_gaba_b,
            self.V_PV, self.V_SST, self.V_VIP
        ) = fused_layer_step(
            self.V, self.phase, self.spikes, self.refractory_timer, self.ATP, self.is_dead,
            self.s_gaba_a, self.s_gaba_b,
            self.V_PV, self.V_SST, self.V_VIP,
            self.I_exc, self.I_ext, self.I_mirror,
            self.chem.dopamine.to(self.V.device), self.chem.acetylcholine.to(self.V.device), self.chem.propofol_conc.to(self.V.device),
            global_context,
            dt,
            self.p_cfg.tau_mem, self.p_cfg.v_rest, self.p_cfg.v_threshold, self.p_cfg.refractory_period,
            self.p_cfg.c_spike, self.p_cfg.c_recovery, self.p_cfg.c_baseline, self.p_cfg.critical_atp,
            self.chem.cfg.tau_gaba_a, self.chem.cfg.tau_gaba_b, self.chem.get_gaba_conductance_modifier().to(self.V.device),
            self.p_cfg.coupling_strength, self.p_cfg.alpha_sync,
            self.natural_omegas,
            self.p_cfg.initial_atp
        )
        
        # Rate tracking
        instant_rate = self.spikes.float().mean() / dt
        self.smoothed_rate += (dt / 0.1) * (instant_rate - self.smoothed_rate)

        # Очистка входов
        self.I_exc.zero_()
        self.I_ext.zero_()
        self.I_mirror.zero_()

    def get_activity_rate(self) -> torch.Tensor:
        return self.smoothed_rate
    
    def get_state_vector(self) -> torch.Tensor:
        return self.V
    
    def validate_state(self) -> bool:
        """Проверка корректности состояния слоя"""
        issues = []
        
        if not torch.all(torch.isfinite(self.V)):
            issues.append("Non-finite values in membrane potential V")
        
        if not torch.all(torch.isfinite(self.phase)):
            issues.append("Non-finite values in phase")
        
        if torch.any(self.ATP < 0):
            issues.append(f"Negative ATP detected: min={self.ATP.min().item():.4f}")
        
        if torch.any(self.ATP > 1.0):
            issues.append(f"ATP exceeds max: max={self.ATP.max().item():.4f}")
        
        if torch.any(self.spikes & self.is_dead):
            issues.append("Dead neurons are spiking!")
        
        if issues:
            print(f"⚠️ Layer {self.id} validation FAILED:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True

    def get_state(self) -> dict:
        return {
            'V': self.V,
            'phase': self.phase,
            'ATP': self.ATP,
            'natural_omegas': self.natural_omegas,
            'spikes': self.spikes,
            'refractory_timer': self.refractory_timer,
            's_gaba_a': self.s_gaba_a,
            's_gaba_b': self.s_gaba_b,
            'smoothed_rate': self.smoothed_rate,
            'is_dead': self.is_dead
        }

    def load_state(self, state: dict):
        device = self.V.device
        for key, value in state.items():
            if hasattr(self, key):
                target = getattr(self, key)
                if isinstance(target, torch.Tensor) and isinstance(value, torch.Tensor):
                    if target.shape == value.shape:
                        target.copy_(value.to(device))
                elif isinstance(target, torch.Tensor) and not isinstance(value, torch.Tensor):
                     target.fill_(value)

# ==========================================

class DynamicNeuralLayer(NeuralLayer):
    """
    Расширение NeuralLayer с поддержкой онтогенеза (изменения размера).
    """
    def __init__(self, layer_id: str, initial_size: int, phys_cfg: PhysicsConfig, chem_sys: BioChemistry):
        # Инициализируем с малым количеством нейронов
        super().__init__(layer_id, initial_size, phys_cfg, chem_sys)
        self.max_capacity = 10000 # Абсолютный лимит для аллокации буферов (если нужно)

    def add_neurons(self, count: int):
        """
        Добавляет 'count' новых нейронов, сохраняя состояние старых.
        """
        current_N = self.N
        new_N = current_N + count
        
        # 1. Расширение тензоров состояния
        # Мы создаем новые тензоры и копируем туда старые данные
        # Это операция GPU-to-GPU, она быстрая, но не мгновенная.
        
        # Helper для ресайза 1D тензоров
        def resize_1d(tensor, fill_value=0.0, std=0.0):
            new_tensor = torch.zeros(new_N, device=DEVICE, dtype=tensor.dtype)
            new_tensor[:current_N] = tensor
            if std > 0:
                new_tensor[current_N:] = torch.normal(mean=fill_value, std=std, size=(count,), device=DEVICE)
            else:
                new_tensor[current_N:] = fill_value
            return new_tensor

        # -- Ресайз переменных --
        self.V = resize_1d(self.V, fill_value=self.p_cfg.v_rest, std=0.1)
        self.phase = resize_1d(self.phase, fill_value=0.0, std=math.pi) # Новые нейроны рассинхронизированы
        
        # Натуральные частоты для новых нейронов
        new_omegas = torch.normal(
            self.p_cfg.omega_base, 
            self.p_cfg.omega_std, 
            size=(count,), 
            device=DEVICE
        ).to(dtype=torch.float64)
        
        self.natural_omegas = torch.cat([self.natural_omegas, new_omegas])
        
        # Остальные буферы
        self.spikes = torch.cat([self.spikes, torch.zeros(count, dtype=torch.bool, device=DEVICE)])
        self.refractory_timer = resize_1d(self.refractory_timer)
        self.s_gaba_a = resize_1d(self.s_gaba_a)
        self.s_gaba_b = resize_1d(self.s_gaba_b)
        
        # Энергия: новые нейроны рождаются с полным запасом ATP
        self.ATP = resize_1d(self.ATP, fill_value=self.p_cfg.initial_atp)
        self.is_dead = torch.cat([self.is_dead, torch.zeros(count, dtype=torch.bool, device=DEVICE)])
        
        # Входные буферы (важно очистить или инициализировать)
        self.I_exc = torch.zeros(new_N, device=DEVICE)
        self.I_ext = torch.zeros(new_N, device=DEVICE)
        self.I_mirror = torch.zeros(new_N, device=DEVICE)
        
        # Интернейроны (масштабируем пропорционально, если нужно, или оставляем фиксированными)
        # Для простоты пока не растим пул интернейронов, либо добавляем логику:
        target_interneurons = int(new_N * 0.2)
        if target_interneurons > self.n_interneurons:
            diff = target_interneurons - self.n_interneurons
            self.V_PV = torch.cat([self.V_PV, torch.zeros(diff, device=DEVICE)])
            self.V_SST = torch.cat([self.V_SST, torch.zeros(diff, device=DEVICE)])
            self.V_VIP = torch.cat([self.V_VIP, torch.zeros(diff, device=DEVICE)])
            self.n_interneurons = target_interneurons

        # Обновляем счетчик
        self.N = new_N
        # print(f"  [Neurogenesis] Layer {self.id} grew to {self.N} neurons.")

# 5. СИНАПТИЧЕСКАЯ ПЛАСТИЧНОСТЬ (STDP + Small-World)
# ==========================================

class SynapseMatrix:
    """
    Управляет матрицей весов между двумя слоями нейронов.
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Векторизованный STDP.
    """
    def __init__(self, n_pre: int, n_post: int, chem_sys: BioChemistry,
                 topo_cfg: Optional[TopologyConfig] = None, layer_type: str = "bottom_up"):
        self.n_pre = n_pre
        self.n_post = n_post
        self.chem = chem_sys
        
        # === ТОПОЛОГИЯ ===
        if topo_cfg and topo_cfg.topology_type == "small_world":
            # Оптимизация памяти: сохраняем маску как bool (1 байт вместо 8)
            raw_mask = TopologyGenerator.create_small_world_mask(n_pre, n_post, topo_cfg, layer_type)
            self.mask = raw_mask.to(device=DEVICE, dtype=torch.bool)  # Экономия 7 байт на синапс!
            
            self.W = torch.abs(torch.randn(n_post, n_pre, device=DEVICE) * math.sqrt(2.0 / n_pre))
            self.W *= self.mask * topo_cfg.local_boost  # Используем self.mask для инициализации весов
            
            # --- 2. Замена плотных матриц (Dense) на разреженные (Sparse CSR) ---
            self.W_dense = self.W * self.mask
            self.W_sparse = self.W_dense.to_sparse_csr()
            self.is_sparse = True
            
            density = self.mask.sum().item() / (n_pre * n_post)
            print(f"  🕸️ {layer_type}: density={density:.3f}, boost={topo_cfg.local_boost:.1f} (Sparse CSR enabled)")
        else:
            self.W = torch.abs(torch.randn(n_post, n_pre, device=DEVICE) * math.sqrt(2.0 / n_pre))
            self.mask = None
            self.is_sparse = False
        
        self.theta_M = torch.ones(n_post, device=DEVICE) * 0.5
        self.activity_history = torch.zeros(n_post, device=DEVICE)
        self.trace_pre = torch.zeros(n_pre, device=DEVICE)
        self.trace_post = torch.zeros(n_post, device=DEVICE)
        self.tau_stdp = 0.02
        self.learning_rate_base = 0.001
        
        # --- 4. Асинхронная пластичность ---
        self.update_interval = 10
        self.step_counter = 0

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        if self.is_sparse:
            # Матричное умножение разреженной матрицы на плотный вектор спайков
            # torch.mv работает намного быстрее для CSR матриц
            return torch.mv(self.W_sparse, pre_spikes.float())
        else:
            W_safe = torch.nan_to_num(self.W, nan=0.0, posinf=5.0, neginf=0.0)
            pre_safe = torch.nan_to_num(pre_spikes.float(), nan=0.0, posinf=1.0, neginf=0.0)
            return torch.matmul(W_safe, pre_safe)

    def backward(self, post_signal: torch.Tensor) -> torch.Tensor:
        """Back-projects signal from post-synaptic to pre-synaptic space (Transpose)"""
        # Note: ignoring sparsity optimization for backward pass for now, using dense W
        if self.is_sparse:
             return torch.matmul(self.W_dense.t(), post_signal.float())
        else:
             return torch.matmul(self.W.t(), post_signal.float())

    @torch.no_grad()
    @torch.no_grad()
    def update_plasticity(self, dt: float, 
                          pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                          pre_phase: torch.Tensor, post_phase: torch.Tensor,
                          sleep_stage: str = 'Wake'):
        """ОПТИМИЗИРОВАНО: Event-Driven STDP + Lazy Sparse Sync"""
        
        if sleep_stage == 'SWS':
            downscale_factor = 0.0001 * dt
            # Optimization: In-place multiplication
            if self.is_sparse:
                 self.W_dense.mul_(1.0 - downscale_factor)
            else:
                 self.W.mul_(1.0 - downscale_factor)
            return

        # 1. Traces (Global update needed for history)
        # Using in-place operations to save memory
        self.trace_pre.mul_(1.0 - dt / self.tau_stdp).add_(pre_spikes.float())
        self.trace_post.mul_(1.0 - dt / self.tau_stdp).add_(post_spikes.float())
        
        # 2. Event-Driven Check: If no post-synaptic spikes, skip expensive matrix ops
        post_spike_indices = torch.nonzero(post_spikes).squeeze(-1)
        if post_spike_indices.numel() == 0:
            return

        # --- 3. Strided / Throttled Updates ---
        self.step_counter += 1
        if self.step_counter % self.update_interval != 0:
            return

        # 4. Sparse/Active-Only Calculation
        # We only need to update rows corresponding to spiking post-neurons
        
        # Gather phases for spiking neurons only [k]
        active_post_phases = post_phase[post_spike_indices]
        
        # Calclulate phase diff for active rows only: [k, 1] - [1, N_pre] -> [k, N_pre]
        phase_diff = active_post_phases.unsqueeze(1) - pre_phase.unsqueeze(0)
        phase_mod = 1.0 + 0.5 * torch.cos(phase_diff)
        
        # Broadcast trace_pre: [1, N_pre]
        # Result delta: [k, N_pre]
        delta_w_active = self.trace_pre.unsqueeze(0) * phase_mod
        
        # [MODIFIED] Bidirectional Dopamine Modulation
        # DA > 0.4 -> LTP (Positive Learning)
        # DA < 0.4 -> LTD (Negative Learning / Avoidance)
        # Pivot is 0.4. Scale factor 5.0 ensures strong reaction.
        da_modulation = (self.chem.dopamine - 0.4) * 5.0
        
        # Base Learning Rate scaled by modulation (can be negative!)
        effective_learning_signal = self.learning_rate_base * da_modulation
        
        # 5. Apply Updates
        if self.is_sparse:
            # We update W_dense but sync to W_sparse lazily
            
            # Extract effective signal for spiking neurons [k]
            # self.theta_M is [N_post], we need [k]
            # Scaling by theta_M (Homeostatic scaling)
            eff_lr_active = effective_learning_signal / (self.theta_M[post_spike_indices] + 0.1)
            
            # Decay factor (Always positive, always reduces weights)
            decay_factor = self.learning_rate_base * 0.1
            
            current_weights_active = self.W_dense[post_spike_indices, :]
            
            # Total delta for active rows:
            # Update = (Signal * Hebbian) - (Decay * CurrentW)
            # If Signal is negative (LTD), Hebbian term reduces weights.
            row_updates = eff_lr_active.unsqueeze(1) * delta_w_active - decay_factor * current_weights_active
            
            # Scatter add to dense matrix
            self.W_dense[post_spike_indices, :] += row_updates
            
            # Clamp in place
            self.W_dense[post_spike_indices, :] = torch.clamp(self.W_dense[post_spike_indices, :], 0.0, 5.0)
            
            # LAZY SYNC: Re-create sparse tensor only occasionally
            if self.step_counter % 100 == 0:
                self.W_dense.masked_fill_(~self.mask, 0.0)
                self.W_sparse = self.W_dense.to_sparse_csr()
                
        else:
            # Dense Case
            eff_lr_active = effective_learning_signal / (self.theta_M[post_spike_indices] + 0.1)
            decay_factor = self.learning_rate_base * 0.1
            
            current_weights_active = self.W[post_spike_indices, :]
            row_updates = eff_lr_active.unsqueeze(1) * delta_w_active - decay_factor * current_weights_active
            
            self.W[post_spike_indices, :] += row_updates
            self.W[post_spike_indices, :] = torch.clamp(self.W[post_spike_indices, :], 0.0, 5.0)

        # 6. Metaplasticity (Only for active neurons)
        sliding_window_alpha = 0.01
        self.activity_history[post_spike_indices] += sliding_window_alpha * (1.0 - self.activity_history[post_spike_indices])
        
        # Optimized Metaplasticity Update
        target_rate = 0.1
        d_theta = 0.001 * (self.activity_history - target_rate)
        self.theta_M = torch.clamp(self.theta_M + d_theta, 0.1, 10.0)

    def get_state(self) -> dict:
        state = {
            'W_dense': self.W_dense if self.is_sparse else self.W,
            'theta_M': self.theta_M,
            'activity_history': self.activity_history,
            'trace_pre': self.trace_pre,
            'trace_post': self.trace_post
        }
        return state

    def load_state(self, state: dict):
        device = self.theta_M.device
        if 'W_dense' in state:
             W_loaded = state['W_dense'].to(device)
             if self.is_sparse:
                 if self.W_dense.shape == W_loaded.shape:
                      self.W_dense.copy_(W_loaded)
                      self.W_sparse = self.W_dense.to_sparse_csr()
             else:
                 if self.W.shape == W_loaded.shape:
                      self.W.copy_(W_loaded)
        
        if 'theta_M' in state: self.theta_M.copy_(state['theta_M'].to(device))
        if 'activity_history' in state: self.activity_history.copy_(state['activity_history'].to(device))
        if 'trace_pre' in state: self.trace_pre.copy_(state['trace_pre'].to(device))
        if 'trace_post' in state: self.trace_post.copy_(state['trace_post'].to(device))

# ==========================================
# 6. ИЕРАРХИЯ И ПРЕДИКТИВНОЕ КОДИРОВАНИЕ
# ==========================================

class DynamicSynapseMatrix(SynapseMatrix):
    def __init__(self, n_pre: int, n_post: int, chem_sys: BioChemistry,
                 topo_cfg: Optional[TopologyConfig] = None, layer_type: str = "bottom_up"):
        super().__init__(n_pre, n_post, chem_sys, topo_cfg, layer_type)
        self.topo_cfg = topo_cfg
        self.layer_type = layer_type

    def resize_matrix(self, new_n_pre: int, new_n_post: int):
        """
        Изменяет размер матрицы весов.
        Старые веса остаются на своих местах (верхний левый угол).
        Новые веса инициализируются согласно топологии.
        """
        if new_n_pre == self.n_pre and new_n_post == self.n_post:
            return

        old_pre = self.n_pre
        old_post = self.n_post
        
        # 1. Создаем новые тензоры
        # Инициализируем нулями (или разреженной структурой)
        new_W_dense = torch.zeros(new_n_post, new_n_pre, device=DEVICE)
        
        # Копируем старый блок
        new_W_dense[:old_post, :old_pre] = self.W_dense if self.is_sparse else self.W
        
        # 2. Инициализация НОВЫХ связей (Neurogenesis Integration)
        # Новые нейроны должны сразу образовывать связи, иначе они бесполезны.
        
        # а) Связи для новых Pre-нейронов (новые столбцы)
        if new_n_pre > old_pre:
            # Генерируем связи для диапазона [old_pre : new_n_pre]
            # Используем упрощенную логику Small-World: локальные связи + случайные дальние
            added_cols = new_n_pre - old_pre
            
            # Локальность: подключаем к случайным нейронам в post-слое
            # Плотность новых связей делаем чуть ниже, чтобы "мягко" войти в сеть
            density = 0.1 
            mask_new_cols = (torch.rand(new_n_post, added_cols, device=DEVICE) < density).float()
            weights_new_cols = torch.abs(torch.randn(new_n_post, added_cols, device=DEVICE) * 0.05)
            
            new_W_dense[:, old_pre:] = weights_new_cols * mask_new_cols

        # б) Связи для новых Post-нейронов (новые строки)
        if new_n_post > old_post:
            added_rows = new_n_post - old_post
            density = 0.1
            mask_new_rows = (torch.rand(added_rows, new_n_pre, device=DEVICE) < density).float()
            weights_new_rows = torch.abs(torch.randn(added_rows, new_n_pre, device=DEVICE) * 0.05)
            
            new_W_dense[old_post:, :] = weights_new_rows * mask_new_rows

        # 3. Обновление мета-параметров STDP
        self.theta_M = self._resize_vector(self.theta_M, new_n_post, fill=0.5)
        self.activity_history = self._resize_vector(self.activity_history, new_n_post, fill=0.0)
        self.trace_pre = self._resize_vector(self.trace_pre, new_n_pre, fill=0.0)
        self.trace_post = self._resize_vector(self.trace_post, new_n_post, fill=0.0)
        
        # 4. Сохранение
        self.n_pre = new_n_pre
        self.n_post = new_n_post
        
        if self.is_sparse:
            self.W_dense = new_W_dense
            # Обновляем маску (для ленивой синхронизации)
            new_mask = (new_W_dense > 0).bool() # Восстанавливаем маску из значений
            self.mask = new_mask
            self.W_sparse = self.W_dense.to_sparse_csr()
        else:
            self.W = new_W_dense
            
    def _resize_vector(self, vec, new_size, fill=0.0):
        if vec.shape[0] == new_size: return vec
        new_vec = torch.ones(new_size, device=DEVICE) * fill
        new_vec[:vec.shape[0]] = vec
        return new_vec


class PredictiveUnit:
    """
    Реализация узла предиктивного кодирования согласно Free Energy Principle.
    
    Математика:
    1. Prediction: μ_l = g(X_{l+1}, a)
    2. Prediction Error: ε_l = Π · (Input - μ_l)
    3. Variational Free Energy: F ≈ 0.5 * ε^T · Π · ε + KL_terms
    """
    def __init__(self, name: str, size: int, phys_cfg: PhysicsConfig, chem_sys: BioChemistry):
        self.layer = NeuralLayer(name, size, phys_cfg, chem_sys)
        
        # Состояния
        self.mu = torch.zeros(size, device=DEVICE)             # Prediction (μ)
        self.prediction_error = torch.zeros(size, device=DEVICE) # Error (ε)
        self.precision = torch.ones(size, device=DEVICE)       # Precision (Π) - обратная дисперсия
        
        # Буферы для асинхронности
        self.input_buffer = torch.zeros(size, device=DEVICE)
        self.top_down_buffer = torch.zeros(size, device=DEVICE)
        
        # Синаптические связи
        self.synapse_bottom_up: Optional[SynapseMatrix] = None
        self.synapse_top_down: Optional[SynapseMatrix] = None
        
        # Интеграция действия (Lateral connection for Efference Copy)
        # Проецирует вектор действия (например, 1D или 2D) в размерность слоя
        self.action_projection = torch.randn(size, 1, device=DEVICE) * 0.1 

    def update_precision(self):
        """
        Обновляет точность (Π) на основе нейромодуляции.
        Π ∝ 1 + [DA] + [ACh] - [NE] (uncertainty)
        """
        da = self.layer.chem.dopamine
        ach = self.layer.chem.acetylcholine
        ne = self.layer.chem.norepinephrine
        
        # NE сигнализирует о неожиданности/неопределенности -> снижает precision priors
        # DA/ACh повышают соотношение сигнал/шум
        base_precision = 1.0 + 1.5 * da + 1.0 * ach - 0.8 * ne
        self.precision = torch.clamp(torch.ones_like(self.precision) * base_precision, 0.1, 10.0)

    def calculate_error_and_free_energy(self, bottom_up_input: torch.Tensor) -> float:
        """
        ε_l(t) = Π_l^{1/2} · [Input(t) - μ_l(t)]
        F = 0.5 * Σ ε_l²
        """
        # 1. Обновляем точность
        self.update_precision()
        
        # 2. Вычисляем расхождение
        raw_error = bottom_up_input - self.mu
        
        # 2.5. КРИТИЧНО: Ограничиваем сырую ошибку для предотвращения взрыва
        # Мембранный потенциал может быть большим (~0-10), но ошибка должна быть разумной
        raw_error = torch.clamp(raw_error, -5.0, 5.0)
        
        # 3. Взвешенная ошибка (Precision-weighted prediction error)
        # Используем квадратный корень из точности для амплитуды ошибки
        self.prediction_error = raw_error * torch.sqrt(self.precision)
        
        # 4. Свободная энергия (аппроксимация через сумму квадратов ошибок)
        # F = 0.5 * (Input - μ)^T * Π * (Input - μ)
        # Нормализуем на количество нейронов для стабильности
        n_neurons = self.prediction_error.shape[0]
        free_energy = 0.5 * torch.sum(self.prediction_error ** 2) / max(n_neurons, 1)
        
        return free_energy

    def update_generative_dynamics(self, dt: float, top_down_input: torch.Tensor, 
                                   action_vector: torch.Tensor,
                                   global_context: Optional[torch.Tensor] = None,
                                   is_dreaming: bool = False):
        """
        Обновление состояния слоя (μ) с учетом:
        1. Top-Down Prior (X_{l+1})
        2. Bottom-Up Error (ε_l, который проталкивается в динамику V)
        3. Action (Efference Copy)
        """
        # --- 1. Формирование предсказания (Generative Function g) ---
        # μ_l ≈ W_td * X_{l+1} + W_action * a(t)
        
        # === Автоматическая адаптация под размер вектора действия ===
        if action_vector.numel() > 0:
            action_dim = action_vector.shape[0]
            # Если размерность изменилась (например, с 1 на 9), пересоздаем матрицу
            if self.action_projection.shape[1] != action_dim:
                self.action_projection = torch.randn(
                    self.layer.N, action_dim,
                    device=self.action_projection.device,
                    dtype=self.action_projection.dtype
                ) * 0.1
        # ===============================================================

        eff_copy = torch.matmul(self.action_projection, action_vector.unsqueeze(1)).squeeze() if action_vector.numel() > 0 else 0.0
        
        prior_drive = top_down_input + eff_copy
        
        # --- 2. Динамика слоя (Update X/μ) ---
        # dμ/dt = -∂F/∂μ = ε * Π + ...
        # В нашей реализации слой NeuralLayer уже имеет динамику dV/dt.
        # Мы подаем ошибку как возбуждающий/корректирующий ток.
        
        if is_dreaming:
            # Во сне отключаем сенсорную коррекцию, полагаемся на prior + шум
            drive = (prior_drive - self.layer.V) + torch.randn_like(self.layer.V) * 0.5
        else:
            # Бодрствование: баланс между Prior и Error
            # Если ошибка положительная (вход > ожидания), мы должны повысить активность
            correction = self.prediction_error * torch.sqrt(self.precision)
            drive = correction + (prior_drive - self.layer.V)

        self.layer.I_ext += drive
        
        # Шаг физики нейронов
        self.layer.dynamics_step(dt, global_context=global_context)
        
        # Текущее состояние слоя становится предсказанием (μ) для уровня ниже
        self.mu = self.layer.get_state_vector()

    def get_state(self) -> dict:
        state = {
            'layer': self.layer.get_state(),
            'mu': self.mu,
            'precision': self.precision,
            'action_projection': self.action_projection
        }
        if self.synapse_bottom_up:
            state['synapse_bottom_up'] = self.synapse_bottom_up.get_state()
        if self.synapse_top_down:
            state['synapse_top_down'] = self.synapse_top_down.get_state()
        return state

    def load_state(self, state: dict):
        device = self.mu.device
        if 'layer' in state:
            self.layer.load_state(state['layer'])
        if 'mu' in state: self.mu.copy_(state['mu'].to(device))
        if 'precision' in state: self.precision.copy_(state['precision'].to(device))
        if 'action_projection' in state: 
             proj = state['action_projection'].to(device)
             if self.action_projection.shape == proj.shape:
                 self.action_projection.copy_(proj)
             else:
                 self.action_projection = proj
                 
        if 'synapse_bottom_up' in state and self.synapse_bottom_up:
            self.synapse_bottom_up.load_state(state['synapse_bottom_up'])
        if 'synapse_top_down' in state and self.synapse_top_down:
            self.synapse_top_down.load_state(state['synapse_top_down'])

class HippocampalSystem:
    def __init__(self, capacity: int = 1000, replay_strength: float = 5.0):
        self.capacity = capacity
        self.buffer: List[torch.Tensor] = []
        self.replay_strength = replay_strength
        
    def store(self, state: torch.Tensor, valence: float):
        # Запоминаем только эмоционально окрашенные события
        if abs(valence) > 0.3:
            self.buffer.append(state.detach().clone())
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)
    
    def replay_sws(self, target_layer: NeuralLayer, dt: float) -> bool:
        """Впрыскивает воспоминание в кору во время медленного сна"""
        if len(self.buffer) == 0: return False
        
        # Вероятность рипппла (SWR) на каждом шаге
        if torch.rand(1).item() < 0.05:
            idx = torch.randint(0, len(self.buffer), (1,)).item()
            memory = self.buffer[idx]
            
            # Подгоняем размерность
            if memory.shape[0] != target_layer.N:
                min_dim = min(memory.shape[0], target_layer.N)
                current_input = torch.zeros(target_layer.N, device=DEVICE)
                current_input[:min_dim] = (memory[:min_dim] - target_layer.V[:min_dim])
            else:
                current_input = (memory - target_layer.V)
            
            # Инъекция тока
            target_layer.I_ext += current_input * self.replay_strength
            return True
        return False


class BrainHierarchy:
    """
    Управляет иерархией предиктивных блоков.
    Обеспечивает передачу сообщений (Bottom-Up Errors, Top-Down Predictions, Action Context).
    """
    def __init__(self, phys_cfg: PhysicsConfig, chem_sys: BioChemistry, 
                 topo_cfg: Optional[TopologyConfig] = None, input_dim: int = 2000):
        # OPTIMIZED CONFIGURATION for RTX 4070 (~62000 neurons total)
        sizes = [input_dim, 4000, 3000, 2000] 
        names = ["V1_Sensory", "V2_Association", "IT_Object", "PFC_Executive"]
        
        self.levels: List[PredictiveUnit] = []
        
        # Input Projection (Optional, defaults to Identity/None if input matches V1)
        self.input_projection: Optional[SynapseMatrix] = None
        
        for name, size in zip(names, sizes):
            unit = PredictiveUnit(name, size, phys_cfg, chem_sys)
            self.levels.append(unit)
            
        self.connections = []
        for i in range(len(self.levels) - 1):
            lower = self.levels[i]
            higher = self.levels[i+1]
            
            # Связи Small-World
            bu_syn = SynapseMatrix(lower.layer.N, higher.layer.N, chem_sys, topo_cfg, "bottom_up")
            lower.synapse_bottom_up = bu_syn 
            
            td_syn = SynapseMatrix(higher.layer.N, lower.layer.N, chem_sys, topo_cfg, "top_down")
            higher.synapse_top_down = td_syn
            
            self.connections.append((bu_syn, td_syn))

    def process_sensory_input(self, sensory_input: torch.Tensor, 
                              action_vector: torch.Tensor,
                              dt: float, 
                              sleep_stage: str, 
                              global_context: Optional[torch.Tensor] = None):
        """
        Полный цикл обновления иерархии:
        1. Расчет ошибок (Bottom-Up Pass)
        2. Динамика состояний (Top-Down + Lateral Action Pass)
        3. Передача сообщений между слоями
        """

        is_dreaming = (sleep_stage == 'REM')
        scaling_factor = 1.0

        # === ФАЗА 1: РАСЧЕТ ОШИБОК И СВОБОДНОЙ ЭНЕРГИИ ===
        
        # Apply Input Projection if exists
        if self.input_projection:
             v1_input = self.input_projection.forward(sensory_input)
        else:
             v1_input = sensory_input

        # Сначала нижний уровень сравнивает вход с прогнозом прошлого шага
        self.levels[0].calculate_error_and_free_energy(v1_input)
        
        # Остальные уровни сравнивают пришедший снизу сигнал (Error Signal) с внутренним прогнозом
        # Примечание: В классическом PC слой L получает ошибку от L-1 и посылает прогноз в L-1.
        # Здесь мы упрощаем: delayed_bottom_up - это проекция ошибки снизу.
        for i in range(1, len(self.levels)):
            unit = self.levels[i]
            unit.calculate_error_and_free_energy(unit.input_buffer)

        # === ФАЗА 2: ОБНОВЛЕНИЕ ГЕНЕРАТИВНОЙ МОДЕЛИ ===
        # Обновляем μ (V) на основе priors, errors и efference copy
        for i in range(len(self.levels)):
            unit = self.levels[i]
            
            # PFC (верхний уровень) не имеет входа сверху, используем 0 или memory
            if i == len(self.levels) - 1:
                prior = torch.zeros(unit.layer.N, device=DEVICE)
            else:
                prior = unit.top_down_buffer

            unit.update_generative_dynamics(
                dt=dt, 
                top_down_input=prior,
                action_vector=action_vector, # Передаем копию действия
                global_context=global_context,
                is_dreaming=is_dreaming
            )

        # === ФАЗА 3: ПЕРЕДАЧА СООБЩЕНИЙ (МЕЖСЛОЙНАЯ КОММУНИКАЦИЯ) ===
        
        # 3.1 Top-Down Predictions (Сверху вниз)
        for i in range(len(self.levels) - 1, -1, -1):
            if i < len(self.levels) - 1:
                higher_unit = self.levels[i+1]
                td_syn = higher_unit.synapse_top_down
                
                # Прогноз посылается вниз
                signal = td_syn.forward(higher_unit.mu.float())
                self.levels[i].top_down_buffer = signal * scaling_factor

        # 3.2 Bottom-Up Errors (Снизу вверх)
        # Мы передаем ошибку (prediction error), а не само состояние!
        for i in range(len(self.levels) - 1):
            lower_unit = self.levels[i]
            bu_syn = self.connections[i][0] 
            
            # Ошибка проецируется вверх
            signal = bu_syn.forward(lower_unit.prediction_error)
            self.levels[i+1].input_buffer = signal * scaling_factor

        # === ФАЗА 4: ПЛАСТИЧНОСТЬ ===
        if sleep_stage != 'SWS': # Во время SWS пластичность отключается или меняется (Hebb renormalization)
            for i in range(len(self.connections)):
                bu, td = self.connections[i]
                lower = self.levels[i].layer
                higher = self.levels[i+1].layer
                
                # Обучение на минимизацию ошибки
                bu.update_plasticity(dt, lower.spikes, higher.spikes, lower.phase, higher.phase, sleep_stage)
                td.update_plasticity(dt, higher.spikes, lower.spikes, higher.phase, lower.phase, sleep_stage)
            
    def get_global_free_energy(self) -> float:
        total = 0.0
        total_neurons = 0
        for l in self.levels:
            # Сумма квадратов взвешенных ошибок
            n_neurons = l.prediction_error.shape[0]
            total += torch.sum(l.prediction_error ** 2).item()
            total_neurons += n_neurons
        # Нормализуем на общее количество нейронов
        return total / max(total_neurons, 1)

    def get_sensory_prediction_error(self) -> torch.Tensor:
        """Returns the prediction error projected back to sensory space"""
        v1_error = self.levels[0].prediction_error
        if self.input_projection:
            return self.input_projection.backward(v1_error)
        else:
            return v1_error

    def get_state(self) -> dict:
        return {
            'levels': [level.get_state() for level in self.levels]
        }
        
    def load_state(self, state: dict):
        if 'levels' in state:
            for i, level_state in enumerate(state['levels']):
                if i < len(self.levels):
                    self.levels[i].load_state(level_state)

class NeurogenesisManager:
    def __init__(self, hierarchy: BrainHierarchy, config: NeurogenesisConfig):
        self.hierarchy = hierarchy
        self.cfg = config
        self.step_counter = 0
        
        # Статус развития
        self.phase = "MORPHOGENESIS" # or "MATURATION"
        
        print(f"🧬 Neurogenesis System Online. Target Base: {self.cfg.base_target_neurons}")

    def update(self, free_energy: float):
        self.step_counter += 1
        if self.step_counter % self.cfg.update_interval != 0:
            return

        # Проходим по всем уровням иерархии
        for i, unit in enumerate(self.hierarchy.levels):
            layer = unit.layer # Это должен быть DynamicNeuralLayer
            
            # --- Логика решений ---
            growth_needed = 0
            
            # 1. Проверка OOM
            if layer.N >= self.cfg.max_neurons_per_layer:
                continue

            # 2. Определение фазы и необходимости роста
            if layer.N < self.cfg.base_target_neurons:
                # ФАЗА 1: Быстрый рост
                self.phase = "MORPHOGENESIS"
                growth_needed = self.cfg.growth_rate_fast
            else:
                # ФАЗА 2: Адаптивный рост
                self.phase = "MATURATION"
                
                # Растем, только если ошибка высокая (нужны новые ресурсы для объяснения мира)
                # И если достаточно энергии (ATP)
                avg_atp = layer.ATP.mean().item()
                if free_energy > self.cfg.error_threshold_trigger and avg_atp > self.cfg.atp_cost_per_neuron:
                    growth_needed = self.cfg.growth_rate_slow
                    # print(f"  [Growth] Layer {layer.id} triggered adaptation (+{growth_needed}) due to FE={free_energy:.2f}")

            # 3. Выполнение роста
            if growth_needed > 0:
                print(f"  [Neurogenesis] Layer '{layer.id}' growing: {layer.N} -> {layer.N + growth_needed} (Reason: {self.phase}, FE={free_energy:.2f})")
                self._execute_growth(unit, i, growth_needed)

    def _execute_growth(self, unit, level_idx, count):
        # 1. Растим сам слой (нейроны)
        # Приводим к типу DynamicNeuralLayer для IDE, по факту он там и есть
        if isinstance(unit.layer, DynamicNeuralLayer):
            unit.layer.add_neurons(count)
        
        # 2. Ресайзим буферы предиктивного кодирования
        new_size = unit.layer.N
        
        # Ресайз mu, error, precision
        def resize_vec(v):
            new_v = torch.zeros(new_size, device=DEVICE)
            new_v[:v.shape[0]] = v
            return new_v
            
        unit.mu = resize_vec(unit.mu)
        unit.prediction_error = resize_vec(unit.prediction_error)
        unit.precision = resize_vec(unit.precision)
        unit.precision[unit.precision == 0] = 1.0 # Новые нейроны имеют базовую точность
        
        unit.input_buffer = resize_vec(unit.input_buffer)
        unit.top_down_buffer = resize_vec(unit.top_down_buffer)
        
        # Ресайз action projection
        if unit.action_projection.numel() > 0:
            act_dim = unit.action_projection.shape[1]
            new_proj = torch.randn(new_size, act_dim, device=DEVICE) * 0.1
            new_proj[:unit.action_projection.shape[0], :] = unit.action_projection
            unit.action_projection = new_proj

        # 3. Ресайзим Синапсы (Corrected Logic)
        
        # A) Исходящие связи (Unit -> Outputs)
        # Unit является PRE-синаптическим для своих .synapse_bottom_up и .synapse_top_down
        
        # Unit -> Higher (Bottom-Up Output)
        if unit.synapse_bottom_up and isinstance(unit.synapse_bottom_up, DynamicSynapseMatrix):
            # Pre=Unit(Grew), Post=Higher(Fixed)
            # print(f"    -> Resizing Output (Bottom-Up) to {unit.synapse_bottom_up.n_post}x{new_size}")
            unit.synapse_bottom_up.resize_matrix(new_size, unit.synapse_bottom_up.n_post)

        # Unit -> Lower (Top-Down Output)
        if unit.synapse_top_down and isinstance(unit.synapse_top_down, DynamicSynapseMatrix):
            # Pre=Unit(Grew), Post=Lower(Fixed)
            unit.synapse_top_down.resize_matrix(new_size, unit.synapse_top_down.n_post)

        # B) Входящие связи (Inputs -> Unit)
        # Unit является POST-синаптическим для связей соседей
        
        # Lower -> Unit (Bottom-Up Input)
        if level_idx > 0:
            lower_unit = self.hierarchy.levels[level_idx - 1]
            if lower_unit.synapse_bottom_up and isinstance(lower_unit.synapse_bottom_up, DynamicSynapseMatrix):
                 # Pre=Lower(Fixed), Post=Unit(Grew)
                 lower_unit.synapse_bottom_up.resize_matrix(lower_unit.synapse_bottom_up.n_pre, new_size)
        
        # Higher -> Unit (Top-Down Input)
        if level_idx < len(self.hierarchy.levels) - 1:
            higher_unit = self.hierarchy.levels[level_idx + 1]
            if higher_unit.synapse_top_down and isinstance(higher_unit.synapse_top_down, DynamicSynapseMatrix):
                 # Pre=Higher(Fixed), Post=Unit(Grew)
                 higher_unit.synapse_top_down.resize_matrix(higher_unit.synapse_top_down.n_pre, new_size)

        # 4. Resize Input Projection if V1 grew (Special Case Input)
        if level_idx == 0 and self.hierarchy.input_projection:
             if isinstance(self.hierarchy.input_projection, DynamicSynapseMatrix):
                   # Pre=Sensory(Fixed), Post=V1(Grew)
                   self.hierarchy.input_projection.resize_matrix(
                        self.hierarchy.input_projection.n_pre, 
                        new_size
                   )

class DynamicBrainHierarchy(BrainHierarchy):
    def __init__(self, phys_cfg: PhysicsConfig, chem_sys: BioChemistry, 
                 topo_cfg: Optional[TopologyConfig], neuro_cfg: NeurogenesisConfig):
        
        # ВАЖНО: Стартуем с малого размера (initial_neurons), а не с полного
        init_size = neuro_cfg.initial_neurons
        
        # Имена слоев те же
        names = ["V1_Sensory", "V2_Association", "IT_Object", "PFC_Executive"]
        
        self.levels: List[PredictiveUnit] = []
        
        # Initialize Input Projection (Sensory -> V1)
        # Pre=Sensory(Fixed=2000), Post=V1(Growing=init_size)
        input_dim = 2000 
        self.input_projection = DynamicSynapseMatrix(
            n_pre=input_dim, 
            n_post=init_size, 
            chem_sys=chem_sys,
            topo_cfg=topo_cfg,
            layer_type="bottom_up"
        )
        
        for name in names:
            # Создаем Unit, но внутри подменяем NeuralLayer на DynamicNeuralLayer
            unit = PredictiveUnit(name, init_size, phys_cfg, chem_sys)
            # Переопределение слоя
            unit.layer = DynamicNeuralLayer(name, init_size, phys_cfg, chem_sys)
            
            # Переопределение action_projection под новый размер
            unit.action_projection = torch.randn(init_size, 1, device=DEVICE) * 0.1
            
            # Ресайз внутренних буферов unit'а под init_size
            unit.mu = torch.zeros(init_size, device=DEVICE)
            unit.prediction_error = torch.zeros(init_size, device=DEVICE)
            unit.precision = torch.ones(init_size, device=DEVICE)
            unit.input_buffer = torch.zeros(init_size, device=DEVICE)
            unit.top_down_buffer = torch.zeros(init_size, device=DEVICE)
            
            self.levels.append(unit)
            
        self.connections = []
        # Создаем динамические синапсы
        for i in range(len(self.levels) - 1):
            lower = self.levels[i]
            higher = self.levels[i+1]
            
            # Используем DynamicSynapseMatrix
            bu_syn = DynamicSynapseMatrix(lower.layer.N, higher.layer.N, chem_sys, topo_cfg, "bottom_up")
            lower.synapse_bottom_up = bu_syn 
            
            td_syn = DynamicSynapseMatrix(higher.layer.N, lower.layer.N, chem_sys, topo_cfg, "top_down")
            higher.synapse_top_down = td_syn
            
            self.connections.append((bu_syn, td_syn))

    def load_state(self, state: dict):
        # 1. Resize layers to match saved state
        if 'levels' in state:
            for i, level_state in enumerate(state['levels']):
                if i >= len(self.levels): break
                
                # Deduce saved size
                target_size = -1
                if 'mu' in level_state:
                     target_size = level_state['mu'].shape[0]
                elif 'layer' in level_state and 'V' in level_state['layer']: 
                     target_size = level_state['layer']['V'].shape[0]
                
                current_size = self.levels[i].layer.N
                
                if target_size > current_size:
                    print(f"📥 Loading Brain: Resizing Layer {i} ({current_size} -> {target_size})")
                    diff = target_size - current_size
                    self._resize_level(level_idx=i, count=diff)
        
        # 2. Standard Load
        super().load_state(state)

    def _resize_level(self, level_idx, count):
        unit = self.levels[level_idx]
        
        # 1. Grow Neurons
        if hasattr(unit.layer, 'add_neurons'):
             unit.layer.add_neurons(count)
             
        new_size = unit.layer.N
        
        # 2. Resize Predictive Buffers
        def resize_vec(v):
            if v.shape[0] == new_size: return v
            new_v = torch.zeros(new_size, device=DEVICE)
            min_len = min(v.shape[0], new_size)
            new_v[:min_len] = v[:min_len]
            return new_v
            
        unit.mu = resize_vec(unit.mu)
        unit.prediction_error = resize_vec(unit.prediction_error)
        unit.precision = resize_vec(unit.precision)
        unit.precision[unit.precision == 0] = 1.0 
        
        unit.input_buffer = resize_vec(unit.input_buffer)
        unit.top_down_buffer = resize_vec(unit.top_down_buffer)
        
        # Action Projection
        if unit.action_projection.numel() > 0:
            act_dim = unit.action_projection.shape[1]
            new_proj = torch.randn(new_size, act_dim, device=DEVICE) * 0.1
            min_len = min(unit.action_projection.shape[0], new_size)
            new_proj[:min_len, :] = unit.action_projection[:min_len, :]
            unit.action_projection = new_proj

        # 3. Resize Synapses
        # Output Bottom-Up
        if unit.synapse_bottom_up and hasattr(unit.synapse_bottom_up, 'resize_matrix'):
            unit.synapse_bottom_up.resize_matrix(new_size, unit.synapse_bottom_up.n_post)
            
        # Output Top-Down
        if unit.synapse_top_down and hasattr(unit.synapse_top_down, 'resize_matrix'):
            unit.synapse_top_down.resize_matrix(new_size, unit.synapse_top_down.n_post)
            
        # Input Bottom-Up (from Lower)
        if level_idx > 0:
            lower = self.levels[level_idx - 1]
            if lower.synapse_bottom_up and hasattr(lower.synapse_bottom_up, 'resize_matrix'):
                 lower.synapse_bottom_up.resize_matrix(lower.synapse_bottom_up.n_pre, new_size)
                 
        # Input Top-Down (from Higher)
        if level_idx < len(self.levels) - 1:
            higher = self.levels[level_idx + 1]
            if higher.synapse_top_down and hasattr(higher.synapse_top_down, 'resize_matrix'):
                 higher.synapse_top_down.resize_matrix(higher.synapse_top_down.n_pre, new_size)
                 
        # Input Projection (V1)
        if level_idx == 0 and self.input_projection:
             if hasattr(self.input_projection, 'resize_matrix'):
                  self.input_projection.resize_matrix(self.input_projection.n_pre, new_size)

# ==========================================
# 7. ГЛОБАЛЬНОЕ РАБОЧЕЕ ПРОСТРАНСТВО (GWT)
# ==========================================

class GlobalWorkspace:
    def __init__(self, hierarchy: BrainHierarchy):
        self.hierarchy = hierarchy
        self.theta_activity = 5.0
        self.theta_sync = 0.8      
        self.capacity = 4
        
        self.active_coalitions: List[str] = [] 
        self.broadcast_signal = torch.zeros(100, device=DEVICE)
        self.phi_current = 0.0

        # ID областей + Social + Agency
        self.area_ids = [u.layer.id for u in self.hierarchy.levels] + ['Social_Self', 'Sense_of_Agency']
        # ОПТИМИЗАЦИЯ: кэш индексов для O(1) поиска вместо O(n) list.index()
        self.area_id_to_idx = {name: i for i, name in enumerate(self.area_ids)}
        n = len(self.area_ids)
        # Случайная инициализация (в идеале должна обучаться)
        self.semantic_matrix = torch.rand(n, n, device=DEVICE)
        self.semantic_matrix = (self.semantic_matrix + self.semantic_matrix.T) / 2.0

    def step(self, dt: float, social_module: Optional['SocialCognition'] = None):
        candidates = []
        
        # 1. Сбор кандидатов с нормализацией активности
        for unit in self.hierarchy.levels:
            activity = unit.layer.get_activity_rate().item()
            R, _ = unit.layer.compute_kuramoto_order()
            R_val = R.item()
            
            # === Логарифмическая чувствительность (Weber-Fechner) ===
            norm_activity = math.log1p(activity / 10.0) * 5.0

            # Условие входа: активность выше фона и есть синхронизация
            if activity > self.theta_activity and R_val > self.theta_sync:
                candidates.append({
                    'id': unit.layer.id,
                    'score': norm_activity * R_val,
                    'data': unit.layer.get_state_vector()
                })
        
        # 2. Социальный контекст
        if social_module is not None:
            social_pain = social_module.get_social_pain_signal()
            social_urgency = social_pain * 20.0 

            # Получаем уровень пропофола из первого слоя иерархии
            propofol = self.hierarchy.levels[0].layer.chem.propofol_conc
            
            # Вычисляем "фактор бодрствования" (1.0 -> 0.0)
            awake_factor = torch.clamp(1.0 - (propofol / 3.0), 0.0, 1.0)
            
            # Глушим социальную срочность анестезией
            social_urgency *= awake_factor
            
            if social_urgency > 0.5:
                social_vector = social_module.m2_self_in_other
                pad_size = 100 - social_vector.shape[0]
                if pad_size > 0:
                    padded_social = torch.cat([social_vector, torch.zeros(pad_size, device=social_vector.device)])
                else:
                    padded_social = social_vector[:100]
                
                candidates.append({
                    'id': 'Social_Self_Model',
                    'score': social_urgency,
                    'data': padded_social
                })

        # Семантическое усиление - ОПТИМИЗИРОВАНО: используем dict.get() вместо list.index()
        if self.active_coalitions:
            for cand in candidates:
                binding_boost = 0.0
                cand_idx = self.area_id_to_idx.get(cand['id'], -1)
                
                if cand_idx >= 0:
                    for active_id in self.active_coalitions:
                        act_idx = self.area_id_to_idx.get(active_id, -1)
                        if act_idx >= 0:
                            binding_boost += self.semantic_matrix[cand_idx, act_idx].item()
                
                cand['score'] += binding_boost * 2.0

        # 3. Конкуренция
        candidates.sort(key=lambda x: x['score'], reverse=True)
        winners = candidates[:self.capacity]
        self.active_coalitions = [w['id'] for w in winners]
        
        # 4. Глобальное вещание и Интеграция
        target_phi = 0.0
        
        if winners:
            winner_tensors = [w['data'][:100] for w in winners]
            stacked = torch.stack(winner_tensors)
            combined_signal = torch.mean(stacked, dim=0)
            
            self.broadcast_signal = self.broadcast_signal * 0.9 + combined_signal * 0.1
            target_phi = sum(w['score'] for w in winners)
        else:
            self.broadcast_signal *= 0.9 
            target_phi = 0.0
            
        # === ИНЕРЦИЯ СОЗНАНИЯ (экспоненциальное сглаживание) ===
        tau_ignition = 0.250

        alpha = 1.0 - torch.exp(torch.tensor(-dt / tau_ignition))
        alpha = float(alpha.item())

        self.phi_current += alpha * (target_phi - self.phi_current)

    def get_context_feedback(self) -> torch.Tensor:
        return self.broadcast_signal

    def get_state(self) -> dict:
        return {
            'semantic_matrix': self.semantic_matrix
        }
    
    def load_state(self, state: dict):
        if 'semantic_matrix' in state:
            self.semantic_matrix.copy_(state['semantic_matrix'].to(self.semantic_matrix.device))

# ==========================================
# 8. ЭМОЦИОНАЛЬНАЯ СИСТЕМА
# ==========================================

class AffectiveSystem:
    def __init__(self, chem: BioChemistry):
        self.chem = chem
        
        self.valence = torch.tensor(0.0, device=DEVICE) 
        self.arousal = torch.tensor(0.0, device=DEVICE) 
        
        self.amygdala_activity = torch.tensor(0.0, device=DEVICE) 
        self.insula_activity = torch.tensor(0.0, device=DEVICE)    
        self.vmPFC_activity = torch.tensor(0.0, device=DEVICE)     
        
        self.tau_valence = 5.0  
        self.tau_arousal = 2.0
        
    def update(self, dt: float, 
               total_free_energy: float, 
               body_pain: float, 
               reward_signal: float,
               simulation_time: float = 0.0):
        # Защита от аномально высоких значений (после нормализации должны быть ~0.01-1.0)
        if total_free_energy > 10.0:
            effective_free_energy = 0.0
        else:
            effective_free_energy = total_free_energy

        # 1. Amygdala
        target_amygdala = 0.8 * body_pain + 0.5 * math.tanh(effective_free_energy)
        self.vmPFC_activity = 0.5 * (1.0 - self.arousal) 
        effective_threat = target_amygdala * (1.0 - 0.5 * self.vmPFC_activity)
        
        self.amygdala_activity += dt * (effective_threat - self.amygdala_activity)

        # 2. Arousal
        drive_arousal = (
            2.0 * math.tanh(effective_free_energy) +  
            1.0 * self.amygdala_activity +          
            0.5 * self.chem.norepinephrine          
        )
        dA = (1.0 / self.tau_arousal) * (-self.arousal + drive_arousal)
        self.arousal = torch.clamp(self.arousal + dA * dt, 0.0, 1.0)

        # 3. Valence
        expected_reward = 0.0 
        rpe = reward_signal - expected_reward
        
        drive_valence = (
            1.0 * rpe                       
            - 1.0 * body_pain               
            - 0.5 * effective_free_energy   
        )
        
        dV = (1.0 / self.tau_valence) * (-self.valence + drive_valence)
        self.valence = torch.clamp(self.valence + dV * dt, -1.0, 1.0)
        
        # 4. Insula
        self.insula_activity = 0.7 * body_pain + 0.3 * torch.abs(self.valence)

    def get_modulation_factors(self) -> dict:
        return {
            'learning_rate_mod': 1.0 + 0.5 * self.arousal.item(),
            'attention_precision': 1.0 + 1.0 * self.amygdala_activity.item(),
            'gamma_freq_shift': 5.0 * self.arousal.item() 
        }

    def get_state(self) -> dict:
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'amygdala': self.amygdala_activity,
            'insula': self.insula_activity
        }
    
    def load_state(self, state: dict):
        self.valence = state.get('valence', self.valence)
        self.arousal = state.get('arousal', self.arousal)
        self.amygdala_activity = state.get('amygdala', self.amygdala_activity)
        self.insula_activity = state.get('insula', self.insula_activity)

# ==========================================
# 9. SOCIAL COGNITION
# ==========================================

@dataclass
class MentalModel:
    intentions: torch.Tensor
    beliefs: torch.Tensor
    emotional_state: torch.Tensor 

class SocialCognition:
    """
    Theory of Mind: Рекурсивная модель M0, M1, M2.
    Раздел 19 математической модели.
    """
    def __init__(self, vector_dim: int = 10):
        self.dim = vector_dim
        
        # M0: Модель себя
        self.m0_intentions = torch.zeros(self.dim, device=DEVICE)
        self.m0_beliefs = torch.zeros(self.dim, device=DEVICE)
        self.m0_emotions = torch.zeros(2, device=DEVICE)
        
        # M1: Модель другого
        self.m1_other_intentions = torch.zeros(self.dim, device=DEVICE)
        self.m1_other_beliefs = torch.zeros(self.dim, device=DEVICE)
        self.m1_other_emotions = torch.zeros(2, device=DEVICE)
        
        # M2: Мета-модель ("что он думает обо мне")
        self.m2_self_in_other = torch.zeros(self.dim, device=DEVICE)
        
        self.inference_rate = 0.1

    def update_self(self, my_actions: torch.Tensor, my_valence: float, my_arousal: float):
        """Обновление M0 на основе собственных действий и эмоций"""
        if my_actions.shape[0] > self.dim:
            my_actions = my_actions[:self.dim]
        elif my_actions.shape[0] < self.dim:
            pad = torch.zeros(self.dim - my_actions.shape[0], device=my_actions.device)
            my_actions = torch.cat([my_actions, pad])
        
        self.m0_intentions += 0.1 * (my_actions - self.m0_intentions)
        self.m0_emotions = torch.tensor([my_valence, my_arousal], device=DEVICE)

    def observe_other(self, observed_behavior: torch.Tensor, dt: float):
        """
        Inverse inference: наблюдая поведение другого, делаем вывод о его намерениях.
        Раздел 19b математической модели.
        """
        if observed_behavior.shape[0] > self.dim:
            observed_behavior = observed_behavior[:self.dim]
        elif observed_behavior.shape[0] < self.dim:
            pad = torch.zeros(self.dim - observed_behavior.shape[0], device=observed_behavior.device)
            observed_behavior = torch.cat([observed_behavior, pad])
        
        # Предсказание поведения на основе текущей модели
        predicted_behavior = self.m1_other_intentions * 0.8 + self.m1_other_beliefs * 0.2
        prediction_error = observed_behavior - predicted_behavior
        
        # Обновление модели другого (Inverse RL)
        self.m1_other_intentions += self.inference_rate * prediction_error * dt
        
        # Упрощённый вывод эмоций по энергии поведения
        behavior_energy = torch.norm(observed_behavior)
        inferred_arousal = torch.tanh(behavior_energy)
        self.m1_other_emotions[1] = inferred_arousal

    def recursive_update(self, dt: float):
        """
        M2: "Что он думает обо мне?"
        Раздел 19e математической модели.
        """
        # Расхождение между моим представлением о себе и мета-моделью
        discrepancy = self.m0_intentions - self.m2_self_in_other
        self.m2_self_in_other += 0.05 * discrepancy * dt

    def get_social_pain_signal(self) -> float:
        """Социальная боль от воспринимаемого отвержения"""
        ideal_self = torch.ones(self.dim, device=DEVICE)
        perceived_rejection = torch.norm(ideal_self - self.m2_self_in_other)
        return perceived_rejection.item() * 0.1

    def get_state(self) -> dict:
        return {
            'm0_intentions': self.m0_intentions,
            'm0_beliefs': self.m0_beliefs,
            'm2_self_in_other': self.m2_self_in_other
        }
    
    def load_state(self, state: dict):
        device = self.m0_intentions.device
        if 'm0_intentions' in state: self.m0_intentions.copy_(state['m0_intentions'].to(device))
        if 'm0_beliefs' in state: self.m0_beliefs.copy_(state['m0_beliefs'].to(device))
        if 'm2_self_in_other' in state: self.m2_self_in_other.copy_(state['m2_self_in_other'].to(device))

# ==========================================
# 10. PCI CALCULATOR
# ==========================================

class PCICalculator:
    """
    ОПТИМИЗИРОВАННЫЙ калькулятор PCI.
    Использует сжатие zlib вместо медленного LZ76 для больших массивов.
    """
    @staticmethod
    def compute_pci(spike_matrix: torch.Tensor, window_size: int = None) -> float:
        if window_size is None:
            window_size = spike_matrix.shape[0]
            
        window = spike_matrix[:window_size, :]
        binary_vec_np = window.flatten().byte().cpu().numpy()
        
        # 1. Проверка на "Тишину" (Silence check)
        # Если спайков слишком мало (< 0.1% активности), сложность считать бессмысленно
        spike_density = binary_vec_np.mean()
        if spike_density < 0.001: 
            return 0.0
            
        data_bytes = binary_vec_np.tobytes()
        L = len(binary_vec_np)
        
        # 2. Сжатие
        compressed = zlib.compress(data_bytes)
        # Вычитаем примерный оверхед заголовка zlib (обычно около 10-12 байт для малых данных), 
        # чтобы убрать шум на низких значениях, или используем Lempel-Ziv напрямую.
        c_lz_approx = max(0, len(compressed)) 
        
        # 3. Энтропия Шеннона
        entropy_h = -spike_density * math.log2(spike_density) - (1 - spike_density) * math.log2(1 - spike_density)
        
        # 4. Нормализация (защита от деления на ноль уже частично есть, но усилим)
        theoretical_min_bits = L * entropy_h
        
        # Если теоретический минимум битов слишком мал (сигнал слишком простой), PCI не определен или 0
        if theoretical_min_bits < 50.0:  # Порог в битах
            return 0.0
        
        pci = (c_lz_approx * 8.0) / theoretical_min_bits
        
        return pci
    @staticmethod
    def apply_tms_pulse(layer, strength: float = 20.0):
        n_focal = int(layer.N * 0.15)
        center = torch.randint(0, layer.N, (1,)).item()
        indices = torch.arange(center - n_focal//2, center + n_focal//2) % layer.N
        pulse = torch.zeros(layer.N, device=DEVICE)
        pulse[indices] = strength * (1.0 + torch.randn(len(indices), device=DEVICE) * 0.2)
        layer.I_ext += pulse
        layer.phase[indices] = 0.0

# ==========================================
# 11. ТЕЛО И АКТИВНЫЙ ВЫВОД
# ==========================================

class BodyAgent:
    """
    Агент Активного Вывода (Active Inference).
    
    Реализует:
    1. Proprioceptive Loop (Reflex): Быстрая коррекция позы.
    2. Active Inference: da/dt = -∂F/∂a (минимизация сенсорной ошибки).
    3. Exploration: Стохастическое добавление шума при высокой неопределенности.
    4. Sense of Agency (SoA) согласно Section 6 математической модели.
    """
    def __init__(self, n_sensors: int, n_actuators: int):
        # Состояние тела
        self.position = torch.tensor([0.0], device=DEVICE)
        self.velocity = torch.tensor([0.0], device=DEVICE)
        
        # Сенсорика
        self.n_sensors = n_sensors
        self.sensory_input = torch.zeros(n_sensors, device=DEVICE)
        
        # Проприоцепция
        self.proprioception_real = torch.zeros(n_actuators, device=DEVICE)
        self.proprioception_pred = torch.zeros(n_actuators, device=DEVICE)
        
        # Action Dynamics
        self.action_val = torch.zeros(n_actuators, device=DEVICE) 
        
        # Sense of Agency (чувство авторства)
        self.sense_of_agency = 1.0
        
        # Exploration parameters
        self.uncertainty_accumulator = 0.0
        
        # [NEW] Override for Lobotomy/God Mode
        self.action_override = None

    def get_sensory_jacobian(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """
        Оценивает матрицу ∂S/∂a. 
        Аппроксимация через градиент сенсорного поля.
        """
        grad = torch.gradient(sensory_input)[0]
        return -grad  

    def update_action(self, dt: float, 
                      sensory_prediction_error: torch.Tensor, 
                      sensory_precision: torch.Tensor,
                      environment_target: float,
                      dopamine_level: float):
        """
        Главный цикл Активного Вывода и расчет SoA.
        """
        
        # 1. Physics Simulation (Environment)
        self.velocity += (self.action_val - 2.0 * self.velocity) * dt 
        self.position += self.velocity * dt
        
        # Генерация сенсорного входа
        true_error = environment_target - self.position
        tuning_curves = torch.linspace(-1, 1, self.n_sensors, device=DEVICE)
        self.sensory_input = torch.exp(-5.0 * ((tuning_curves - true_error) ** 2))
        
        # 2. Proprioceptive Loop (Reflex)
        k_d = 2.0
        reflex_force = -k_d * self.velocity
        
        # 3. Active Inference (Goal-Directed)
        # da/dt = - (Π_s · ε_s) · ∂S/∂a
        jacobian = self.get_sensory_jacobian(self.sensory_input)
        # Fix: Cast to float to match Jacobian (float32) for dot product
        weighted_error = sensory_prediction_error.float() * sensory_precision.float()
        fe_gradient = torch.dot(weighted_error, jacobian.float()).unsqueeze(0)
        
        alpha_action = 5.0 * (1.0 + dopamine_level)
        da_goal = -alpha_action * fe_gradient
        
        # 4. Exploration strategy
        current_uncertainty = torch.mean(torch.abs(sensory_prediction_error))
        self.uncertainty_accumulator = 0.95 * self.uncertainty_accumulator + 0.05 * current_uncertainty
        
        beta_explore = torch.tanh(self.uncertainty_accumulator * 2.0)
        
        if dopamine_level < 0.2: 
            beta_explore += 0.5
            
        noise = torch.randn_like(self.action_val, device=DEVICE) * 2.0
        
        # 5. Final Action Integration
        total_force = (1.0 - beta_explore) * da_goal + reflex_force + beta_explore * noise
        self.action_val += total_force * dt
        self.action_val = torch.clamp(self.action_val, -10.0, 10.0)

        # ==========================================
        # 6. Update Sense of Agency (SoA) - FIXED
        # ==========================================
        # Theory Section 6: SoA(t) = exp(-lambda * C_action)
        # C_action = ||outcome_predicted - outcome_actual||^2
        
        # FIX 1: Use Mean Squared Error (MSE) instead of Sum to handle 1000 sensors dimensionality
        mse_error = torch.mean(sensory_prediction_error**2)
        
        # FIX 2: Tuned lambda (sensitivity). 
        # If lambda is too high, any tiny error kills SoA. 
        # lambda=10.0 means 10% MSE error results in SoA ~ 0.36
        lambda_soa = 0.5
        
        target_soa = torch.exp(-lambda_soa * mse_error)
        
        # FIX 3: Temporal Smoothing (Agency implies temporal integration)
        # SoA shouldn't flicker instantly with noise.
        tau_soa = 0.5 # seconds
        alpha_soa = dt / tau_soa
        
        self.sense_of_agency = self.sense_of_agency + alpha_soa * (target_soa - self.sense_of_agency)
        # self.sense_of_agency = max(0.0, min(self.sense_of_agency, 1.0))
        # Keep as tensor? Check usage. 
        # If sense_of_agency is tensor (which it becomes if target_soa is tensor), max/min require torch.clamp
        if isinstance(self.sense_of_agency, torch.Tensor):
             self.sense_of_agency = torch.clamp(self.sense_of_agency, 0.0, 1.0)
        else:
             self.sense_of_agency = max(0.0, min(self.sense_of_agency, 1.0))

    def get_action_vector(self) -> torch.Tensor:
        if self.action_override is not None:
            return self.action_override.to(DEVICE)
        return self.action_val
    
    def activate_mirror_neurons(self, observed_action: torch.Tensor, target_size: int) -> torch.Tensor:
        input_dim = observed_action.shape[0]
        if input_dim == 0:
            return torch.zeros(target_size, device=DEVICE)
        repeats = (target_size // input_dim) + 1
        extended = observed_action.repeat(repeats)[:target_size]
        signal = extended * 0.5
        return signal
    
# ==========================================
# 12. МЕНЕДЖЕР СНА И БИОРИТМОВ
# ==========================================

class SleepCycleManager:
    def __init__(self, chem: BioChemistry):
        self.chem = chem
        
        self.time_of_day = 0.0 
        self.sleep_pressure = 0.0 
        self.current_stage = 'Wake'
        self.circadian_process = 0.0
        
        self.pressure_buildup_rate = 1.0 / 16.0 
        self.pressure_decay_rate = 1.0 / 8.0    
        
    def update(self, dt_hours: float, current_stress: torch.Tensor):
        self.time_of_day = (self.time_of_day + dt_hours) % 24.0
        
        self.circadian_process = math.sin(2 * math.pi * (self.time_of_day - 8) / 24.0)
        
        if self.current_stage == 'Wake':
            self.sleep_pressure += self.pressure_buildup_rate * dt_hours
        else:
            self.sleep_pressure -= self.pressure_decay_rate * dt_hours
        
        self.sleep_pressure = max(0.0, min(self.sleep_pressure, 2.0))
        
        sleep_drive = self.sleep_pressure - self.circadian_process 
        
        stress_val = current_stress.item() if isinstance(current_stress, torch.Tensor) else current_stress
        
        if self.current_stage == 'Wake':
            if sleep_drive > 1.2: 
                self.current_stage = 'SWS'
        
        elif self.current_stage == 'SWS':
            if self.sleep_pressure < 0.8 and torch.rand(1).item() < 0.05:
                self.current_stage = 'REM'
            elif self.sleep_pressure < 0.1: 
                self.current_stage = 'Wake'
                
        elif self.current_stage == 'REM':
            if torch.rand(1).item() < 0.1: 
                self.current_stage = 'SWS'

        self.apply_stage_effects()

    def apply_stage_effects(self):
        if self.current_stage == 'Wake':
            self.chem.acetylcholine = torch.tensor(0.8, device=DEVICE) 
            self.chem.norepinephrine = torch.tensor(0.8, device=DEVICE)
            self.chem.serotonin = torch.tensor(0.6, device=DEVICE)
        elif self.current_stage == 'SWS':
            self.chem.acetylcholine = torch.tensor(0.1, device=DEVICE) 
            self.chem.norepinephrine = torch.tensor(0.3, device=DEVICE)
        elif self.current_stage == 'REM':
            self.chem.acetylcholine = torch.tensor(0.9, device=DEVICE)
            self.chem.norepinephrine = torch.tensor(0.05, device=DEVICE)
            self.chem.serotonin = torch.tensor(0.05, device=DEVICE)

# ==========================================
# 13. GENOME MANAGER (MTS)
# ==========================================

class GenomeManager:
    """
    Управляет жизненным циклом генома: извлечение, мутация, сохранение.
    Реализует концепцию MTS (Metabolic Topological Sporulation).
    """
    def __init__(self, spore_dir="spores"):
        self.spore_dir = spore_dir
        if not os.path.exists(spore_dir):
            os.makedirs(spore_dir)
            
    def extract(self, simulator) -> dict:
        """Извлекает текущий геном (конфигурацию) из симулятора."""
        # 1. Chemistry DNA
        chemistry_dna = {
            "tau_mem": float(simulator.phys_cfg.tau_mem),
            "learning_rate": 0.01, 
        }
        
        # 2. Topology DNA
        topology_dna = {
            "type": "Dynamic" if "Neurogenesis" in simulator.__class__.__name__ else "Fixed",
            "layers": [
                {"id": i, "size": level.layer.N} 
                for i, level in enumerate(simulator.hierarchy.levels)
            ]
        }
        
        return {
            "generation": getattr(simulator, "gen", 0),
            "chemistry": chemistry_dna,
            "topology": topology_dna,
            "parent_id": str(id(simulator))
        }

    def mutate(self, genome: dict) -> dict:
        """Applies mutations to the genome."""
        child = json.loads(json.dumps(genome)) # Deep copy
        child["generation"] += 1
        
        # 1. Parametric Drift (80% chance)
        if torch.rand(1).item() < 0.8:
            drift = 1.0 + (torch.rand(1).item() - 0.5) * 0.1 # +/- 5%
            child["chemistry"]["tau_mem"] *= drift
            
        # 2. Synaptic Rewiring (15% chance)
        if torch.rand(1).item() < 0.15:
            child["mutation_log"] = child.get("mutation_log", []) + ["Rewiring"]
            
        return child

    def save(self, genome: dict, filename: str):
        path = os.path.join(self.spore_dir, filename)
        with open(path, 'w') as f:
            json.dump(genome, f, indent=2)
        return path

# ==========================================
# 14. ГЛАВНЫЙ СИМУЛЯТОР
# ==========================================

class ConsciousnessSimulator:
    def __init__(self, use_small_world: bool = True):
        print("Initializing Mathematical Model of Consciousness")
        print(f"Topology: {'Small-World' if use_small_world else 'Random'}")
        
        self.phys_cfg = PhysicsConfig()
        self.chem_cfg = ChemistryConfig()
        self.topo_cfg = TopologyConfig() if use_small_world else None
        
        self.chemistry = BioChemistry(self.chem_cfg)
        self.sleep_manager = SleepCycleManager(self.chemistry)
        self.genome_manager = GenomeManager()
        self.gen = 0

        self.hippocampus = HippocampalSystem()
        
        # Increase input dimension for specialized specialized text/visual embedding
        self.hierarchy = BrainHierarchy(self.phys_cfg, self.chemistry, self.topo_cfg, input_dim=2000)
        self.gwt = GlobalWorkspace(self.hierarchy)
        
        self.affect = AffectiveSystem(self.chemistry)
        self.social = SocialCognition()
        
        self.body = BodyAgent(n_sensors=2000, n_actuators=1) 
        
        self.environment_target = 0.5
        self.simulation_time = 0.0
        self.has_reproduced = False

    def inject_pathology(self, condition: str):
        """
        Injects pathological states into the system.
        Conditions: 'Anesthesia_Propofol', 'Depression', 'Social_Anxiety'
        """
        print(f"!!! INJECTING PATHOLOGY: {condition} !!!")
        
        if condition == "Anesthesia_Propofol":
            self.chemistry.propofol_conc = torch.tensor(3.0, device=DEVICE) 
            
        elif condition == "Depression":
            self.chemistry.dopamine = torch.tensor(0.1, device=DEVICE)
            self.chemistry.serotonin = torch.tensor(0.1, device=DEVICE)
            self.affect.valence = torch.tensor(-0.8, device=DEVICE)
            
        elif condition == "Social_Anxiety":
            # Initialize m2 if not already present or modify existing
            self.social.m2_self_in_other = torch.ones(10, device=DEVICE) * -1.0 
            self.affect.amygdala_activity = torch.tensor(0.9, device=DEVICE)

            self.affect.amygdala_activity = torch.tensor(0.9, device=DEVICE)

    def biological_lifecycle(self):
        """
        Реализует MTS (Metabolic Topological Sporulation).
        Триггер размножения: SWS + High ATP + Low Stress.
        """
        # 1. Проверка условий размножения (во время сна)
        if self.sleep_manager.current_stage == 'SWS':
            
            # Энергия (average across all layers being > 0.95 means huge surplus)
            total_atp = 0.0
            for level in self.hierarchy.levels:
                total_atp += level.layer.ATP.mean().item()
            avg_atp = total_atp / len(self.hierarchy.levels)
            
            # Стресс (low norepinephrine)
            stress_val = self.chemistry.norepinephrine.item()
            
            if avg_atp > 0.95 and stress_val < 0.4:
                # === АКТ РАЗМНОЖЕНИЯ ===
                print(f"🧬 SPORULATION EVENT INITIATED (ATP={avg_atp:.2f}, Stress={stress_val:.2f})")
                
                try:
                    # А. Извлечение ДНК
                    parent_dna = self.genome_manager.extract(self)
                    
                    # Б. Мутация (NEAT)
                    child_dna = self.genome_manager.mutate(parent_dna)
                    
                    # В. Выброс споры (сохранение JSON)
                    filename = f"gen_{self.gen + 1}_child_{id(self)}_{int(self.simulation_time)}.json"
                    saved_path = self.genome_manager.save(child_dna, filename)
                    
                    # Г. Энергетическая плата (Жертва)
                    for level in self.hierarchy.levels:
                        level.layer.ATP.fill_(0.1) # Drain to 10%
                    
                    self.chemistry.dopamine.fill_(0.0)
                    
                    print(f"✅ Spore saved to {saved_path}. Parent exhausted (ATP -> 0.1).")
                    self.has_reproduced = True
                    
                except Exception as e:
                    print(f"❌ Sporulation Failed: {e}")

    @torch.no_grad()
    def step(self, dt: float, time_scale: float = 1.0):
        self.simulation_time += dt
        
        # === ХРОНО-УСКОРИТЕЛЬ ===
        # Мы умножаем dt на time_scale только для SleepManager.
        # Для нейронов dt остается маленьким (0.05), чтобы математика не ломалась.
        scaled_dt_hours = (dt * time_scale) / 3600.0
        
        self.sleep_manager.update(dt_hours=scaled_dt_hours, 
                                current_stress=self.affect.arousal)

        # 1. Получаем текущее действие
        action_vec = self.body.get_action_vector()

        # 2. Получаем параметры от V1 для тела (Active Inference loop)
        v1_error = self.hierarchy.get_sensory_prediction_error()
        v1_precision = self.hierarchy.levels[0].precision.mean() # Scalar approximation
        
        # 3. Обновляем Тело и Среду (Sensory + Action update)
        if self.sleep_manager.current_stage == 'SWS':
            # В SWS сенсорика отключена
            self.body.sensory_input.fill_(0.0)
        elif self.sleep_manager.current_stage == 'REM':
            # В REM сенсорика стохастическая (сны)
            self.body.sensory_input = torch.normal(0, 0.5, size=(self.body.n_sensors,))
        else:
            # Wake: Active Inference
            self.body.update_action(
                dt=dt,
                sensory_prediction_error=v1_error,
                sensory_precision=v1_precision,
                environment_target=self.environment_target,
                dopamine_level=self.chemistry.dopamine.item()
            )
        
        sensory_input = self.body.sensory_input
        
        # 2. Get Global Context (Echo from previous step)
        global_feedback = self.gwt.get_context_feedback()
        
        # 3. Update BioChemistry
        total_free_energy = self.hierarchy.get_global_free_energy()
        self.chemistry.update(dt, 
                            stress_level=self.affect.arousal,
                            reward_prediction_error=0.0)
        
        # 4. Hippocampal Operations
        if self.sleep_manager.current_stage == 'SWS':
            self.hippocampus.replay_sws(self.hierarchy.levels[-1].layer, dt)
        else:
            if hasattr(self, 'gwt'):
                self.hippocampus.store(self.gwt.broadcast_signal, self.affect.valence.item())

        # 5. Process Hierarchy with Global Feedback AND Action Context
        self.hierarchy.process_sensory_input(
            sensory_input=sensory_input,
            action_vector=action_vec,    # <-- Передаем действие
            dt=dt, 
            sleep_stage=self.sleep_manager.current_stage,
            global_context=global_feedback
        )
        
        # 6. Global Workspace Step
        self.gwt.step(dt, social_module=self.social)
        
        # 7. Social Cognition Update
        self.social.update_self(action_vec, self.affect.valence.item(), self.affect.arousal.item())
        self.social.recursive_update(dt)
        
        # 8. MTS Biological Lifecycle (Sporulation Check)
        self.biological_lifecycle()

    def save_brain(self, path: str):
        """Сохраняет состояние всего мозга в файл"""
        print(f"Saving brain state to {path}...")
        state = {
            'hierarchy': self.hierarchy.get_state(),
            'chemistry': self.chemistry.get_state(),
            'affect': self.affect.get_state(),
            'gwt': self.gwt.get_state(),
            'social': self.social.get_state(),
            'simulation_time': self.simulation_time
        }
        torch.save(state, path)
        print("Brain saved successfully.")

    def load_brain(self, path: str):
        """Загружает состояние мозга из файла"""
        if not os.path.exists(path):
            print(f"Brain file {path} not found. Starting fresh.")
            return
            
        print(f"Loading brain state from {path}...")
        try:
            state = torch.load(path, map_location=DEVICE)
            
            if 'hierarchy' in state: self.hierarchy.load_state(state['hierarchy'])
            if 'chemistry' in state: self.chemistry.load_state(state['chemistry'])
            if 'affect' in state: self.affect.load_state(state['affect'])
            if 'gwt' in state: self.gwt.load_state(state['gwt'])
            if 'social' in state: self.social.load_state(state['social'])
            if 'simulation_time' in state: self.simulation_time = state['simulation_time']
            
            print("Brain loaded successfully.")
        except Exception as e:
            print(f"Error loading brain: {e}")

        # 11. Safety Checks (Periodic)
        if self.simulation_time % 10.0 < dt:
            for level in self.hierarchy.levels:
                if not level.layer.validate_state():
                    print(f"❌ Critical Error at T={self.simulation_time:.2f}s")
                    
        # 12. Жизненный цикл (MTS) - Вызываем в конце, чтобы эффект exhaustion сохранился
        self.biological_lifecycle()

    def print_status(self):
        stage = self.sleep_manager.current_stage
        phi = self.gwt.phi_current
        n_coalitions = len(self.gwt.active_coalitions)
        val = self.affect.valence.item()
        arous = self.affect.arousal.item()
        
        # Получаем среднее значение действия для отображения
        action_display = self.body.get_action_vector().mean().item()
        
        print(f"T={self.simulation_time:.2f}s | Stg: {stage} | "
              f"Phi: {phi:.3f} | Coalitions: {n_coalitions} | "
              f"Emo: V={val:.2f}/A={arous:.2f} | "
              f"Action: {action_display:.2f}")

class SimulatorWithNeurogenesis(ConsciousnessSimulator):
    def __init__(self, use_small_world: bool = True):
        print("Initializing Dynamic Neurogenesis Model")
        
        self.phys_cfg = PhysicsConfig()
        self.chem_cfg = ChemistryConfig()
        self.topo_cfg = TopologyConfig() if use_small_world else None
        
        # --- CONFIGURATION OF GROWTH ---
        self.neuro_cfg = NeurogenesisConfig()
        
        self.chemistry = BioChemistry(self.chem_cfg)
        self.sleep_manager = SleepCycleManager(self.chemistry)
        self.hippocampus = HippocampalSystem()
        
        # Используем Динамическую Иерархию
        self.hierarchy = DynamicBrainHierarchy(self.phys_cfg, self.chemistry, self.topo_cfg, self.neuro_cfg)
        
        # Менеджер Нейрогенеза
        self.gardener = NeurogenesisManager(self.hierarchy, self.neuro_cfg)
        self.genome_manager = GenomeManager() # <--- Added missing manager
        
        self.gwt = GlobalWorkspace(self.hierarchy)
        self.affect = AffectiveSystem(self.chemistry)
        self.social = SocialCognition()
        self.body = BodyAgent(n_sensors=2000, n_actuators=1) 
        
        self.environment_target = 0.5
        self.simulation_time = 0.0
        self.has_reproduced = False

    def step(self, dt: float, time_scale: float = 1.0):
        # 1. Обычный шаг физики (вызовет super().step(dt) если бы структура наследовалась идеально, 
        # но мы переопределили иерархию, так что копируем логику step)
        
        # ... (Код шага аналогичен базовому классу) ...
        # Вставляем вызов менеджера роста ПЕРЕД физикой или ПОСЛЕ
        
        current_fe = self.hierarchy.get_global_free_energy()
        
        # Вызываем Садовника (Gardener) для проверки условий роста
        self.gardener.update(current_fe)
        
        # Далее стандартный цикл:
        super().step(dt, time_scale=time_scale)
        
    def print_status(self):
        super().print_status()
        # Добавляем инфо о размере мозга
        sizes = [u.layer.N for u in self.hierarchy.levels]
        print(f"   🧬 Brain Size: {sizes} | Phase: {self.gardener.phase}")

# ==========================================
# 15. ВИЗУАЛЬНАЯ ДЕМОНСТРАЦИЯ (Unified)
# ==========================================

def run_demo_with_plots(
    dt: float = 0.05,
    total_steps: int = 800,
    anesthesia_step: int = 400,
    experiment_mode: str = "ANESTHESIA",
    psychiatry_mode: Optional[str] = None,
    warmup_steps: int = 20,
    use_small_world: bool = True
):
    sim = ConsciousnessSimulator(use_small_world=use_small_world)

    print("=== NeuralBiocore Unified: Simulation Demo ===")
    print(f"Topology: {'Small-World' if use_small_world else 'Random'}")
    print(f"Mode: {experiment_mode}, Steps: {total_steps}, dt: {dt}s")

    # --- WARMUP PROGRESS BAR ---
    # leave=False заставит бар исчезнуть после завершения, чтобы не засорять вывод
    for _ in tqdm(range(warmup_steps), desc="🔥 Warming up", leave=False, ncols=100):
        sim.step(dt)
    
    print(f"Warmup complete. T={sim.simulation_time:.2f}s")

    # Pathologies
    if psychiatry_mode == "DEPRESSION":
        sim.inject_pathology("Depression")
    elif psychiatry_mode == "MANIA":
        sim.chemistry.dopamine = torch.tensor(0.9, device=DEVICE)      
        sim.chemistry.norepinephrine = torch.tensor(0.8, device=DEVICE) 
        sim.affect.valence = torch.tensor(0.8, device=DEVICE)          

    # Data Collection
    times = []
    phi_values = []
    valences = []
    arousals = []
    propofol_levels = []
    v1_activity = []
    v1_sync = []
    atp_levels = []      
    soa_values = []      
    dead_neurons = []    
    sleep_stages_history = []

    # --- MAIN LOOP PROGRESS BAR ---
    for step in tqdm(range(total_steps), desc=f"🧠 Simulating ({experiment_mode})", ncols=100):
        current_time = sim.simulation_time

        # Scenario: Anesthesia
        if experiment_mode == "ANESTHESIA":
            if step >= anesthesia_step:
                sim.chemistry.propofol_conc += 0.02 
                if sim.chemistry.propofol_conc > 5.0:
                    sim.chemistry.propofol_conc = torch.tensor(5.0, device=DEVICE)

        # Scenario: Awakening
        elif experiment_mode == "AWAKENING":
            if 15.0 <= current_time < 45.0:
                sim.chemistry.propofol_conc += 0.02
                if sim.chemistry.propofol_conc > 4.0: sim.chemistry.propofol_conc = torch.tensor(4.0, device=DEVICE)
            elif current_time >= 45.0:
                sim.chemistry.propofol_conc *= 0.95

        # Scenario: Dreams
        elif experiment_mode == "REM_DREAMS":
            if 10.0 <= sim.simulation_time < 30.0:
                sim.sleep_manager.current_stage = 'REM'
            elif 30.0 <= sim.simulation_time < 40.0:
                 sim.sleep_manager.current_stage = 'Wake'

        sim.step(dt)

        # Record Metrics
        times.append(sim.simulation_time)
        phi_values.append(float(sim.gwt.phi_current))
        valences.append(sim.affect.valence.item())
        arousals.append(sim.affect.arousal.item())
        propofol_levels.append(sim.chemistry.propofol_conc.item())
        sleep_stages_history.append(sim.sleep_manager.current_stage)

        v1_layer = sim.hierarchy.levels[0].layer
        v1_activity.append(v1_layer.get_activity_rate().item())
        R, _ = v1_layer.compute_kuramoto_order()
        v1_sync.append(R.item())
        
        atp_levels.append(v1_layer.ATP.mean().item())
        dead_neurons.append(v1_layer.is_dead.sum().item())
        val_soa = sim.body.sense_of_agency
        if isinstance(val_soa, torch.Tensor):
            val_soa = val_soa.item()
        soa_values.append(val_soa)

    # Plotting (код графиков остался без изменений)
    times_np = np.array(times)
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # 1. Consciousness & Propofol
    ax = axes[0]
    ax.plot(times_np, phi_values, label="Φ (Integrated Info)", color="tab:blue", linewidth=2)
    ax.set_ylabel("Φ / Consciousness")
    ax2 = ax.twinx()
    ax2.plot(times_np, propofol_levels, label="[Propofol]", color="tab:red", linestyle="--")
    ax2.set_ylabel("[Propofol] µM")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title(f"Simulation Mode: {experiment_mode} (Small-World: {use_small_world})")

    # 2. Emotions
    ax = axes[1]
    ax.plot(times_np, valences, label="Valence (Pleasure)", color="tab:green")
    ax.plot(times_np, arousals, label="Arousal (Energy)", color="tab:orange")
    ax.set_ylabel("Affect")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 3. Energy & Death
    ax = axes[2]
    ax.plot(times_np, atp_levels, label="ATP Level (V1 Mean)", color="gold", linewidth=2)
    ax.set_ylabel("Metabolic Energy")
    ax.set_ylim(0, 1.1)
    if max(dead_neurons) > 0:
        ax3 = ax.twinx()
        ax3.plot(times_np, dead_neurons, label="Dead Neurons", color="black", linestyle=":")
        ax3.set_ylabel("Count Dead")
        ax3.legend(loc="lower right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 4. Sense of Agency
    ax = axes[3]
    ax.plot(times_np, soa_values, label="Sense of Agency", color="purple")
    ax.set_ylabel("SoA (0-1)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 5. Neural Dynamics
    ax = axes[4]
    ax.plot(times_np, v1_activity, label="Firing Rate", color="tab:purple")
    ax.plot(times_np, v1_sync, label="Synchronization (R)", color="tab:brown")
    ax.set_ylabel("Neural Dynamics")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")

    # Sleep Stage Backgrounds
    stage_colors = {"Wake": "white", "SWS": "#e6e6fa", "REM": "#ffe4e1"}
    current_st = sleep_stages_history[0]
    start_t = times_np[0]
    for i, st in enumerate(sleep_stages_history):
        if st != current_st:
            for ax_i in axes:
                ax_i.axvspan(start_t, times_np[i], color=stage_colors.get(current_st, "white"), alpha=0.5)
            current_st = st
            start_t = times_np[i]
    for ax_i in axes:
        ax_i.axvspan(start_t, times_np[-1], color=stage_colors.get(current_st, "white"), alpha=0.5)

    plt.tight_layout()
    plt.show()


# ==========================================
# 15. LIFECYCLE DEMO
# ==========================================

def run_sleep_lifecycle_demo(
    total_hours: float = 24.0,
    dt_minutes: float = 5.0,
):
    print("=== Sleep/Wake Cycle Demo (Unified) ===")
    
    chem_cfg = ChemistryConfig()
    chemistry = BioChemistry(chem_cfg)
    sleep_manager = SleepCycleManager(chemistry)

    n_steps = int(total_hours * 60.0 / dt_minutes)
    dt_hours = dt_minutes / 60.0

    times_h = []
    stages = []
    sleep_pressures = []
    circadian_values = []
    ach_levels = []
    ne_levels = []
    serotonins = []

    # --- LIFECYCLE PROGRESS BAR ---
    for _ in tqdm(range(n_steps), desc="🌙 Cycling Circadian Rhythms", ncols=100):
        current_hour = sleep_manager.time_of_day
        stress = 0.3 if 8 <= current_hour <= 20 else 0.05
        
        sleep_manager.update(dt_hours=dt_hours, current_stress=torch.tensor(stress, device=DEVICE))

        times_h.append(sleep_manager.time_of_day)
        stages.append(sleep_manager.current_stage)
        sleep_pressures.append(sleep_manager.sleep_pressure)
        circadian_values.append(sleep_manager.circadian_process)
        ach_levels.append(chemistry.acetylcholine.item())
        ne_levels.append(chemistry.norepinephrine.item())
        serotonins.append(chemistry.serotonin.item())

    # Plotting
    times_h_np = np.array(times_h)
    
    stage_map = {"Wake": 2, "REM": 1, "SWS": 0}
    stage_numeric = np.array([stage_map.get(s, np.nan) for s in stages])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.step(times_h_np, stage_numeric, where="post", color="tab:blue")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["SWS", "REM", "Wake"])
    ax.set_title("Circadian Lifecycle")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times_h_np, sleep_pressures, label="Sleep pressure S(t)", color="tab:red")
    ax.plot(times_h_np, circadian_values, label="Circadian process C(t)", color="tab:green")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(times_h_np, ach_levels, label="ACh", color="tab:orange")
    ax.plot(times_h_np, ne_levels, label="NE", color="tab:purple")
    ax.plot(times_h_np, serotonins, label="5-HT", color="tab:brown")
    ax.set_xlabel("Hours")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 16. PCI EXPERIMENT (Hardened Physics)
# ==========================================

def run_pci_experiment(dt: float = 0.001, use_small_world: bool = True):
    print(f"=== PCI Experiment: Fast Optimized (Small-World: {use_small_world}) ===")
    
    conditions = [
        ("Wake", 0.4, 0.0, 0.8, 0.8),
        ("Light_Anesthesia", 0.15, 2.0, 0.4, 0.4),
        ("Deep_Anesthesia", 0.02, 5.0, 0.1, 0.1)
    ]
    
    results = {}
    
    for condition_name, coupling, propofol, ach, ne in tqdm(conditions, desc="🔬 Total Progress", ncols=100):
        tqdm.write(f"\n--- Condition: {condition_name} ---")
        
        sim = ConsciousnessSimulator(use_small_world=use_small_world)
        sim.phys_cfg.coupling_strength = coupling 
        sim.phys_cfg.alpha_sync = 0.3              
        sim.chemistry.propofol_conc = torch.tensor(propofol, device=DEVICE)
        sim.chemistry.acetylcholine = torch.tensor(ach, device=DEVICE) 
        sim.chemistry.norepinephrine = torch.tensor(ne, device=DEVICE)
        
        if propofol > 0:
            for lvl in sim.hierarchy.levels:
                lvl.layer.p_cfg.v_threshold *= (1.0 + 0.1 * propofol)

        # 1. Stabilizing
        for _ in tqdm(range(500), desc="  🔄 Stabilizing", leave=True, ncols=80):
            noise = 1.0 if condition_name == "Wake" else 0.2
            for unit in sim.hierarchy.levels:
                unit.layer.I_ext += torch.randn(unit.layer.N) * noise
            sim.step(dt)
        
        # 2. TMS Pulse
        target_layer = sim.hierarchy.levels[2].layer 
        tqdm.write(f"  ⚡ Pulse -> {target_layer.id}")
        PCICalculator.apply_tms_pulse(target_layer, strength=60.0)
        
        # 3. Recording
        recording_window_ms = 400
        steps = int(recording_window_ms / (dt * 1000))
        all_spikes = []
        
        for t in tqdm(range(steps), desc="  📉 Recording", leave=True, ncols=80):
            bg_noise = 0.1
            for unit in sim.hierarchy.levels:
                unit.layer.I_ext += torch.randn(unit.layer.N) * bg_noise
            sim.step(dt)
            combined_spikes = torch.cat([u.layer.spikes for u in sim.hierarchy.levels])
            all_spikes.append(combined_spikes)
            
        # 4. PCI Calculation
        spike_matrix = torch.stack(all_spikes)
        analysis_matrix = spike_matrix[5:, :] if spike_matrix.shape[0] > 10 else spike_matrix
        
        pci = PCICalculator.compute_pci(analysis_matrix)
        
        total_spikes = analysis_matrix.sum().item()
        active_neurons = (analysis_matrix.sum(dim=0) > 0).sum().item()
        
        tqdm.write(f"  📊 PCI: {pci:.4f} | Spikes: {int(total_spikes)}")
        
        results[condition_name] = {
            'pci': pci,
            'spike_count': int(total_spikes),
            'active_neurons': active_neurons
        }

    print("\n=== FINAL RESULTS ===")
    print(f"{'Condition':<20} {'PCI':>8} {'Spikes':>10}")
    print("-" * 40)
    for c, metrics in results.items():
        print(f"{c:<20} {metrics['pci']:>8.4f} {metrics['spike_count']:>10}")

if __name__ == "__main__":
    # Uncomment the function you wish to run:
    
    # 1. Visual Simulation (Wake/Anesthesia) with Small-World Topology
    run_demo_with_plots(use_small_world=True, experiment_mode="ANESTHESIA")
    
    # 2. Sleep Cycle Analysis
    # run_sleep_lifecycle_demo()
    
    # 3. PCI Complexity Experiment
    # run_pci_experiment(use_small_world=True)