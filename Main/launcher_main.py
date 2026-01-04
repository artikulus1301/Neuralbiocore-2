import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import argparse

# Import core systems
from Neuralbiocore_U_for_GPU import (
    ConsciousnessSimulator as CoreSimulator,
    PCICalculator,
    PhysicsConfig,
    ChemistryConfig,
    TopologyConfig
)

# Import qualia layer
from u_qualia import (
    SimulatorWithQualia,
    QualiaProjector_Advanced,
    QualiaState
)

# Import social cognition
from theory_of_mind import (
    AdvancedSocialCognition,
    AgentSimulator,
    SocialModel
)

# torch.set_default_dtype(torch.float64) # DISABLED: Use Mixed Precision for 8GB GPU

# ==========================================
# UNIFIED CONSCIOUSNESS SYSTEM
# ==========================================

class UnifiedConsciousnessSystem:
    """
    Интегрированная система сознания, объединяющая:
    - Нейробиологическое ядро (Neuralbiocore_U)
    - Феноменологический слой (u_qualia)
    - Социальное познание (theory_of_mind)
    """
    
    def __init__(self, 
                 use_small_world: bool = True,
                 enable_qualia: bool = True,
                 enable_social: bool = True,
                 n_agents: int = 1):
        
        print("=" * 60)
        print("UNIFIED CONSCIOUSNESS SYSTEM - INITIALIZATION")
        print("=" * 60)
        
        self.enable_qualia = enable_qualia
        self.enable_social = enable_social
        self.n_agents = n_agents
        
        # === CORE BRAIN SIMULATOR ===
        print("🧠 Initializing Neural Core...")
        if enable_qualia:
            self.core = SimulatorWithQualia()
            print("   ✓ Qualia Layer: ACTIVE")
        else:
            self.core = CoreSimulator(use_small_world=use_small_world)
            print("   ✓ Standard Neural Core")
        
        # === QUALIA PROJECTOR (if separate from core) ===
        self.qualia_projector = None
        if enable_qualia and not isinstance(self.core, SimulatorWithQualia):
            self.qualia_projector = QualiaProjector_Advanced(gwt_dim=100)
            print("   ✓ External Qualia Projector: ACTIVE")
        
        # === SOCIAL COGNITION ===
        self.social_agents: List[AdvancedSocialCognition] = []
        if enable_social and n_agents > 0:
            print(f"👥 Initializing {n_agents} Social Agent(s)...")
            for i in range(n_agents):
                agent = AdvancedSocialCognition(vector_dim=10)
                self.social_agents.append(agent)
            print(f"   ✓ {n_agents} Social Cognition Module(s): ACTIVE")
        
        # === DATA LOGGING ===
        self.history = {
            'time': [],
            'phi': [],
            'qualia_richness': [],
            'qualia_phi': [],
            'qualia_coherence': [],
            'qualia_self_rec': [],
            'valence': [],
            'arousal': [],
            'social_pain': [],
            'soa': [],
            'atp': [],
            'propofol': [],
            'sleep_stage': []
        }
        
        # === MULTI-AGENT DATA ===
        if n_agents > 1:
            self.agent_histories = [{
                'M0_intentions': [],
                'M1_other': [],
                'M2_meta': [],
                'prediction_error': []
            } for _ in range(n_agents)]
        
        print("=" * 60)
        print("✓ SYSTEM READY")
        print("=" * 60)
    
    def step(self, dt: float):
        """Единый шаг интегрированной системы"""
        
        # 1. CORE NEURAL DYNAMICS
        self.core.step(dt)
        
        # 2. QUALIA COMPUTATION (if enabled)
        current_qualia: Optional[QualiaState] = None
        if self.enable_qualia:
            if isinstance(self.core, SimulatorWithQualia):
                current_qualia = self.core.current_qualia
            elif self.qualia_projector is not None:
                gwt_signal = self.core.gwt.broadcast_signal
                val = self.core.affect.valence
                arous = self.core.affect.arousal
                insula = self.core.affect.insula_activity
                
                current_qualia = self.qualia_projector.compute_qualia_dynamics(
                    gwt_signal, val, arous, insula, dt
                )
        
        # 3. SOCIAL COGNITION (if enabled)
        social_pain_total = 0.0
        if self.enable_social:
            # Single agent: interact with self-model
            if self.n_agents == 1:
                agent = self.social_agents[0]
                
                # Update self-model from body actions
                action_vec = self.core.body.get_action_vector()
                agent.update_M0_self(
                    action_vec,
                    self.core.affect.valence.item(),
                    self.core.affect.arousal.item()
                )
                
                # Observe "other" (dummy or environment feedback)
                dummy_obs = torch.randn(10) * 0.3
                agent.infer_M1_other(dummy_obs)
                
                # Update meta-model (recursion)
                agent.update_M2_recursion(action_vec)
                
                social_pain_total = agent.calculate_social_pain()
            
            # Multi-agent: pairwise interaction
            elif self.n_agents > 1:
                for i, agent in enumerate(self.social_agents):
                    # Each agent observes others' actions
                    for j, other in enumerate(self.social_agents):
                        if i != j:
                            other_action = other.M0.intentions
                            agent.infer_M1_other(other_action)
                    
                    # Update self
                    my_action = agent.M0.intentions + torch.randn(10) * 0.1
                    agent.update_M0_self(my_action, 0.0, 0.5)
                    agent.update_M2_recursion(my_action)
                    
                    social_pain_total += agent.calculate_social_pain()
                
                social_pain_total /= self.n_agents
        
        # 4. CROSS-SYSTEM INTEGRATION
        # Social pain influences chemistry
        if self.enable_social and social_pain_total > 0.1:
            self.core.chemistry.norepinephrine = torch.clamp(
                self.core.chemistry.norepinephrine + dt * social_pain_total,
                0.0, 1.0
            )
        
        # 5. DATA COLLECTION
        self.history['time'].append(self.core.simulation_time)
        self.history['phi'].append(float(self.core.gwt.phi_current))
        self.history['valence'].append(self.core.affect.valence.item())
        self.history['arousal'].append(self.core.affect.arousal.item())
        self.history['social_pain'].append(social_pain_total)
        soa_val = self.core.body.sense_of_agency
        self.history['soa'].append(soa_val.item() if hasattr(soa_val, 'item') else soa_val)
        self.history['atp'].append(self.core.hierarchy.levels[0].layer.ATP.mean().item())
        self.history['propofol'].append(self.core.chemistry.propofol_conc.item())
        self.history['sleep_stage'].append(self.core.sleep_manager.current_stage)
        
        if current_qualia is not None:
            self.history['qualia_richness'].append(current_qualia.richness)
            self.history['qualia_phi'].append(current_qualia.phi_score)
            self.history['qualia_coherence'].append(current_qualia.coherence)
            self.history['qualia_self_rec'].append(current_qualia.self_recognition)
        else:
            self.history['qualia_richness'].append(0.0)
            self.history['qualia_phi'].append(0.0)
            self.history['qualia_coherence'].append(0.0)
            self.history['qualia_self_rec'].append(0.0)
        
        # Multi-agent data
        if self.n_agents > 1:
            for i, agent in enumerate(self.social_agents):
                self.agent_histories[i]['M0_intentions'].append(
                    agent.M0.intentions.mean().item()
                )
                self.agent_histories[i]['M1_other'].append(
                    agent.M1_other.intentions.mean().item()
                )
                self.agent_histories[i]['M2_meta'].append(
                    agent.M2_meta.intentions.mean().item()
                )
                if len(agent.prediction_error_history) > 0:
                    self.agent_histories[i]['prediction_error'].append(
                        agent.prediction_error_history[-1]
                    )
                else:
                    self.agent_histories[i]['prediction_error'].append(0.0)
    
    def inject_pathology(self, condition: str, **kwargs):
        """Инъекция патологии в систему"""
        print(f"⚠️  INJECTING PATHOLOGY: {condition}")
        
        if condition == "ANESTHESIA":
            propofol_level = kwargs.get('propofol', 4.0)
            self.core.chemistry.propofol_conc = torch.tensor(propofol_level)
            print(f"   Propofol: {propofol_level} µM")
        
        elif condition == "DEPRESSION":
            self.core.chemistry.dopamine = torch.tensor(0.1)
            self.core.chemistry.serotonin = torch.tensor(0.1)
            self.core.affect.valence = torch.tensor(-0.8)
            print("   DA↓, 5-HT↓, Valence=-0.8")
        
        elif condition == "SOCIAL_REJECTION":
            if self.enable_social:
                for agent in self.social_agents:
                    agent.M2_meta.intentions = torch.ones(10) * -1.0
                print("   M2 (Self-in-Other) → Negative")
        
        elif condition == "SLEEP_DEPRIVATION":
            self.core.sleep_manager.sleep_pressure = 2.0
            self.core.chemistry.adenosine = torch.tensor(0.9)  # if exists
            print("   Sleep Pressure: MAX")
    
    def print_status(self):
        """Расширенный статус системы"""
        t = self.core.simulation_time
        stage = self.core.sleep_manager.current_stage
        phi = self.core.gwt.phi_current
        
        status_str = f"T={t:.2f}s | Stage: {stage} | Φ: {phi:.3f}"
        
        if self.enable_qualia and len(self.history['qualia_richness']) > 0:
            q_rich = self.history['qualia_richness'][-1]
            q_phi = self.history['qualia_phi'][-1]
            status_str += f" | Q_Rich: {q_rich:.2f}, Q_Φ: {q_phi:.2f}"
        
        if self.enable_social:
            soc_pain = self.history['social_pain'][-1]
            status_str += f" | Social_Pain: {soc_pain:.2f}"
        
        print(status_str)
    
    def visualize(self, save_path: Optional[str] = None):
        """Визуализация всех систем"""
        
        times = np.array(self.history['time'])
        
        # Определяем количество графиков
        n_plots = 5  # Base: Neural, Emotion, Energy, SoA, Sleep
        if self.enable_qualia:
            n_plots += 1
        if self.enable_social and self.n_agents > 1:
            n_plots += 1
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots), sharex=True)
        ax_idx = 0
        
        # 1. CONSCIOUSNESS & PROPOFOL
        ax = axes[ax_idx]
        ax.plot(times, self.history['phi'], label='Φ (GWT)', color='blue', linewidth=2)
        ax.set_ylabel('Consciousness Φ')
        ax2 = ax.twinx()
        ax2.plot(times, self.history['propofol'], label='[Propofol]', color='red', linestyle='--')
        ax2.set_ylabel('[Propofol] µM', color='red')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_title('Unified Consciousness System - Integrated View')
        ax.grid(alpha=0.3)
        ax_idx += 1
        
        # 2. QUALIA (if enabled)
        if self.enable_qualia:
            ax = axes[ax_idx]
            ax.plot(times, self.history['qualia_richness'], label='Richness ||Q||', color='purple')
            ax.plot(times, self.history['qualia_phi'], label='Φ_Q (LZ)', color='magenta')
            ax.plot(times, self.history['qualia_coherence'], label='Coherence', color='cyan')
            ax.plot(times, self.history['qualia_self_rec'], label='Self-Recognition', color='orange', linestyle=':')
            ax.set_ylabel('Qualia Metrics')
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
            ax_idx += 1
        
        # 3. EMOTIONS
        ax = axes[ax_idx]
        ax.plot(times, self.history['valence'], label='Valence', color='green')
        ax.plot(times, self.history['arousal'], label='Arousal', color='orange')
        if self.enable_social:
            ax.plot(times, self.history['social_pain'], label='Social Pain', color='red', linestyle=':')
        ax.set_ylabel('Affective State')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax_idx += 1
        
        # 4. ENERGY & DEATH
        ax = axes[ax_idx]
        ax.plot(times, self.history['atp'], label='ATP (V1 Mean)', color='gold', linewidth=2)
        ax.set_ylabel('Metabolic Energy')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax_idx += 1
        
        # 5. SENSE OF AGENCY
        ax = axes[ax_idx]
        ax.plot(times, self.history['soa'], label='Sense of Agency', color='purple')
        ax.set_ylabel('SoA (0-1)')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax_idx += 1
        
        # 6. MULTI-AGENT SOCIAL DYNAMICS (if enabled)
        if self.enable_social and self.n_agents > 1:
            ax = axes[ax_idx]
            for i, hist in enumerate(self.agent_histories):
                ax.plot(times, hist['M0_intentions'], label=f'Agent {i+1} Self (M0)', linestyle='-')
                ax.plot(times, hist['M1_other'], label=f'Agent {i+1} Other (M1)', linestyle='--', alpha=0.7)
            ax.set_ylabel('Social Intentions')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3)
            ax_idx += 1
        
        # 7. SLEEP STAGES (Background shading)
        stage_colors = {"Wake": "white", "SWS": "#e6e6fa", "REM": "#ffe4e1"}
        if len(self.history['sleep_stage']) > 0:
            current_st = self.history['sleep_stage'][0]
            start_t = times[0]
            for i, st in enumerate(self.history['sleep_stage']):
                if st != current_st or i == len(self.history['sleep_stage']) - 1:
                    end_t = times[i] if i < len(times) else times[-1]
                    for ax in axes:
                        ax.axvspan(start_t, end_t, color=stage_colors.get(current_st, "white"), alpha=0.3)
                    current_st = st
                    start_t = end_t
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved: {save_path}")
        
        plt.show()

# ==========================================
# EXPERIMENT SCENARIOS
# ==========================================

def scenario_consciousness_loss(system: UnifiedConsciousnessSystem, 
                                 dt: float = 0.05, 
                                 steps: int = 600):
    """Сценарий: Постепенная потеря сознания под анестезией"""
    print("\n" + "="*60)
    print("SCENARIO: Consciousness Loss Under Anesthesia")
    print("="*60)
    
    for step in tqdm(range(steps), desc="🧠 Running", ncols=100):
        # Inject propofol gradually after step 200
        if step >= 200:
            current_prop = system.core.chemistry.propofol_conc.item()
            system.core.chemistry.propofol_conc = torch.tensor(
                min(current_prop + 0.01, 5.0)
            )
        
        system.step(dt)
        
        if step % 100 == 0:
            system.print_status()
    
    system.visualize(save_path="consciousness_loss.png")


def scenario_social_interaction(n_agents: int = 2, 
                                 dt: float = 0.05, 
                                 steps: int = 400):
    """Сценарий: Социальное взаимодействие между агентами"""
    print("\n" + "="*60)
    print(f"SCENARIO: Social Interaction ({n_agents} Agents)")
    print("="*60)
    
    system = UnifiedConsciousnessSystem(
        enable_qualia=True,
        enable_social=True,
        n_agents=n_agents
    )
    
    # Inject social stress midway
    for step in tqdm(range(steps), desc="👥 Interacting", ncols=100):
        if step == 200:
            system.inject_pathology("SOCIAL_REJECTION")
        
        system.step(dt)
        
        if step % 100 == 0:
            system.print_status()
    
    system.visualize(save_path="social_interaction.png")


def scenario_dream_state(system: UnifiedConsciousnessSystem,
                          dt: float = 0.05,
                          steps: int = 500):
    """Сценарий: Переход через стадии сна с REM"""
    print("\n" + "="*60)
    print("SCENARIO: Sleep Cycle with REM Dreams")
    print("="*60)
    
    for step in tqdm(range(steps), desc="😴 Dreaming", ncols=100):
        t = system.core.simulation_time
        
        # Force sleep stages
        if 10 <= t < 25:
            system.core.sleep_manager.current_stage = 'SWS'
        elif 25 <= t < 35:
            system.core.sleep_manager.current_stage = 'REM'
        elif t >= 35:
            system.core.sleep_manager.current_stage = 'Wake'
        
        system.step(dt)
        
        if step % 100 == 0:
            system.print_status()
    
    system.visualize(save_path="dream_state.png")


def scenario_qualia_phenomenology(system: UnifiedConsciousnessSystem,
                                    dt: float = 0.05,
                                    steps: int = 400):
    """Сценарий: Изучение феноменологии qualia"""
    print("\n" + "="*60)
    print("SCENARIO: Qualia Phenomenology Study")
    print("="*60)
    
    for step in tqdm(range(steps), desc="🌈 Experiencing", ncols=100):
        t = system.core.simulation_time
        
        # Modulate emotional valence to affect qualia
        if 5 <= t < 10:
            system.core.affect.valence = torch.tensor(0.8)  # Joy
            system.core.affect.arousal = torch.tensor(0.6)
        elif 15 <= t < 20:
            system.core.affect.valence = torch.tensor(-0.7)  # Pain
            system.core.affect.insula_activity = torch.tensor(0.9)
        
        system.step(dt)
        
        if step % 100 == 0:
            system.print_status()
    
    system.visualize(save_path="qualia_phenomenology.png")


def scenario_pci_measurement(dt: float = 0.001, use_small_world: bool = True):
    """Сценарий: Измерение PCI в разных состояниях"""
    print("\n" + "="*60)
    print("SCENARIO: PCI Measurement Across States")
    print("="*60)
    
    conditions = [
        ("Wake", 0.0),
        ("Light Anesthesia", 2.0),
        ("Deep Anesthesia", 5.0)
    ]
    
    results = {}
    
    for cond_name, propofol in conditions:
        print(f"\n--- Testing: {cond_name} ---")
        
        system = UnifiedConsciousnessSystem(
            use_small_world=use_small_world,
            enable_qualia=False,
            enable_social=False
        )
        
        system.core.chemistry.propofol_conc = torch.tensor(propofol)
        
        # Stabilize
        for _ in tqdm(range(500), desc="  Stabilizing", leave=False):
            system.step(dt)
        
        # TMS Pulse
        target_layer = system.core.hierarchy.levels[2].layer
        PCICalculator.apply_tms_pulse(target_layer, strength=60.0)
        
        # Record
        all_spikes = []
        for _ in tqdm(range(400), desc="  Recording", leave=False):
            system.step(dt)
            combined_spikes = torch.cat([
                u.layer.spikes for u in system.core.hierarchy.levels
            ])
            all_spikes.append(combined_spikes)
        
        spike_matrix = torch.stack(all_spikes)
        pci = PCICalculator.compute_pci(spike_matrix[5:, :])
        
        results[cond_name] = pci
        print(f"  PCI: {pci:.4f}")
    
    # Visualize PCI results
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions_list = list(results.keys())
    pci_values = list(results.values())
    
    bars = ax.bar(conditions_list, pci_values, color=['green', 'orange', 'red'])
    ax.set_ylabel('PCI Value')
    ax.set_title('Perturbational Complexity Index Across States')
    ax.axhline(y=0.5, color='black', linestyle='--', label='Consciousness Threshold')
    ax.legend()
    
    for bar, val in zip(bars, pci_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pci_measurement.png', dpi=300)
    plt.show()


# ==========================================
# MAIN LAUNCHER
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Consciousness System Launcher'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='consciousness_loss',
        choices=[
            'consciousness_loss',
            'social_interaction',
            'dream_state',
            'qualia_phenomenology',
            'pci_measurement',
            'custom'
        ],
        help='Select experiment scenario'
    )
    
    parser.add_argument('--dt', type=float, default=0.05, help='Time step (seconds)')
    parser.add_argument('--steps', type=int, default=600, help='Number of steps')
    parser.add_argument('--n_agents', type=int, default=2, help='Number of agents for social scenarios')
    parser.add_argument('--small_world', action='store_true', help='Use small-world topology')
    parser.add_argument('--disable_qualia', action='store_true', help='Disable qualia layer')
    parser.add_argument('--disable_social', action='store_true', help='Disable social cognition')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("UNIFIED CONSCIOUSNESS SYSTEM LAUNCHER")
    print("="*60)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")
    
    # Run selected scenario
    if args.scenario == 'consciousness_loss':
        system = UnifiedConsciousnessSystem(
            use_small_world=args.small_world,
            enable_qualia=not args.disable_qualia,
            enable_social=not args.disable_social
        )
        scenario_consciousness_loss(system, dt=args.dt, steps=args.steps)
    
    elif args.scenario == 'social_interaction':
        scenario_social_interaction(
            n_agents=args.n_agents,
            dt=args.dt,
            steps=args.steps
        )
    
    elif args.scenario == 'dream_state':
        system = UnifiedConsciousnessSystem(
            use_small_world=args.small_world,
            enable_qualia=not args.disable_qualia,
            enable_social=not args.disable_social
        )
        scenario_dream_state(system, dt=args.dt, steps=args.steps)
    
    elif args.scenario == 'qualia_phenomenology':
        system = UnifiedConsciousnessSystem(
            use_small_world=args.small_world,
            enable_qualia=True,  # Force enable
            enable_social=not args.disable_social
        )
        scenario_qualia_phenomenology(system, dt=args.dt, steps=args.steps)
    
    elif args.scenario == 'pci_measurement':
        scenario_pci_measurement(dt=0.001, use_small_world=args.small_world)
    
    elif args.scenario == 'custom':
        # Custom scenario - user can modify this
        system = UnifiedConsciousnessSystem(
            use_small_world=args.small_world,
            enable_qualia=not args.disable_qualia,
            enable_social=not args.disable_social,
            n_agents=args.n_agents
        )
        
        print("\n🛠️  CUSTOM SCENARIO - Running default loop")
        for step in tqdm(range(args.steps), desc="Processing"):
            system.step(args.dt)
            
            if step % 100 == 0:
                system.print_status()
        
        system.visualize(save_path="custom_scenario.png")
    
    print("\n" + "="*60)
    print("✓ SIMULATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Quick test mode if run without arguments
    import sys
    if len(sys.argv) == 1:
        print("Running in QUICK TEST mode...")
        print("Use --help for full options\n")
        
        system = UnifiedConsciousnessSystem(
            use_small_world=True,
            enable_qualia=True,
            enable_social=True,
            n_agents=1
        )
        
        # Quick 200-step demo
        for step in tqdm(range(1000), desc="Quick Demo"):
            if step == 600:
                system.inject_pathology("ANESTHESIA", propofol=3.0)
            
            system.step(0.05)
            
            if step % 50 == 0:
                system.print_status()
        
        system.visualize()
    else:
        main()