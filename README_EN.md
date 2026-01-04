# NeuralBiocore

**NeuralBiocore** is an advanced computational platform for modeling biologically plausible consciousness _in silico_. The project implements the **Mathematical Model of Consciousness (26 points)**, integrating neurobiology, information theory, and cognitive science into a unified architecture.

This project is created not just as a neural network simulation, but as a foundation for **neuromorphic artificial intelligence** capable of phenomenological experience (qualia), social cognition (Theory of Mind), and meta-cognition.

---

## 🚀 Key Features

### 1. Neural Core
*   **Realistic Neurons**: LIF (Leaky Integrate-and-Fire) model with phase dynamics, gamma rhythms (~40 Hz), and refractoriness.
*   **Neurochemistry**: Global and local modulation by neurotransmitters:
    *   **Dopamine (DA)**: Reinforcement learning, prediction precision.
    *   **Acetylcholine (ACh)**: Attention, context switching.
    *   **Serotonin (5-HT)**: Mood regulation.
    *   **Anesthetics (Propofol)**: Modeling loss of consciousness via enhanced GABA inhibition.
*   **Energetics**: Simulation of metabolism (ATP), fatigue, and cellular death upon resource depletion.
*   **Topology**: Small-World networks with modular structure to balance integration and segregation.

### 2. Architecture of Consciousness
*   **Global Workspace Theory (GWT)**: Mechanism for broadcasting information from neuronal coalitions for global availability.
*   **Qualia**: A computational model of subjective experience based on richness, coherence, and self-recognition of states.
*   **Sleep and Dreams**: Full sleep cycles (SWS/REM) for memory consolidation and training generative models.

### 3. Social Cognition (Theory of Mind)
*   **Multi-Agent Simulation**: Support for interaction between multiple agents.
*   **Recursive Modeling**: Agents build models of themselves (M0), models of others (M1), and meta-models (M2 - "what they think about me").
*   **Social Emotions**: Emulation of social pain, empathy, and feelings of rejection.

---

## 📂 Project Structure

*   **`Main/`**: Source code of the system.
    *   `launcher_main.py`: Entry point for running simulation scenarios.
    *   `Neuralbiocore_U_for_GPU.py`: Optimized GPU simulation core (PyTorch JIT).
    *   `u_qualia.py`: Phenomenological consciousness module.
    *   `theory_of_mind.py`: Social cognition module.
*   **`Theory/`**: Theoretical basis of the project.
    *   `Математическая Модель Сознания С 26 пунктом.txt`: Full description of the 26 theory points (in Russian).
*   **`Examples of Experemental results/`**: Examples of experiment visualization (neurodynamics graphs, weights, screenshots).

---

## 🛠️ Installation

Python 3.8+ and a CUDA-capable GPU (recommended) are required.

1.  Clone the repository (if applicable).
2.  Navigate to the `Main` directory:
    ```bash
    cd Main
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Main dependencies: `torch` (2.0+), `numpy`, `matplotlib`, `tqdm`.*

---

## 💻 Usage

Simulations are launched via `launcher_main.py`. You can choose one of the preset scenarios.

### Basic Launch (Loss of Consciousness under Anesthesia)
By default, the propofol injection scenario is launched:
```bash
python launcher_main.py
```

### Available Scenarios (`--scenario`)

1.  **consciousness_loss** (default):
    Simulation of gradual anesthetic injection. GWT breakdown, Φ (Phi) drop, and desynchronization are observed.
    ```bash
    python launcher_main.py --scenario consciousness_loss
    ```

2.  **social_interaction**:
    Launch of multiple agents with Theory of Mind. A "social rejection" event occurs in the middle of the simulation.
    ```bash
    python launcher_main.py --scenario social_interaction --n_agents 2
    ```

3.  **dream_state**:
    Demonstration of sleep cycles. The agent passes through phases of wakefulness, slow-wave sleep (SWS), and rapid eye movement sleep (REM).
    ```bash
    python launcher_main.py --scenario dream_state
    ```

4.  **qualia_phenomenology**:
    Investigation of qualia dynamics under changing emotional backgrounds (joy vs. pain).
    ```bash
    python launcher_main.py --scenario qualia_phenomenology
    ```

5.  **pci_measurement**:
    Measurement of the Perturbational Complexity Index (PCI) using virtual TMS stimulation in different states.
    ```bash
    python launcher_main.py --scenario pci_measurement
    ```

### Command Line Arguments
*   `--steps`: Number of simulation steps (default 600).
*   `--dt`: Time step in seconds (default 0.05).
*   `--disable_qualia`: Disable qualia layer (for speed).
*   `--small_world`: Use Small-World topology (recommended for realism).

---

## 📚 Theoretical Basis (Brief)

The project is based on the **26 Points of the Mathematical Model of Consciousness**, including:

1.  **Phase Dynamics**: Neurons as oscillators synchronizing via phase coupling.
2.  **Hierarchy**: Multi-level processing from sensory to PFC.
3.  **Chemical Modulation**: Influence of neurotransmitters on neuron parameters.
4.  **Plasticity**: STDP accounting for phase and metaplasticity.
5.  **Predictive Coding & Active Inference**: Free Energy minimization.
6.  **Meta-Cognition**: Monitoring errors and confidence (Sense of Agency).
7.  **Global Integration**: Competition of coalitions for access to the workspace.
8.  **Homeostasis**: Energy balance and sleep drive.
9.  **Pathologies**: Modeling schizophrenia, depression, epilepsy.
10. **Criteria of Consciousness**: Integrated Information (IIT) and causal density.
11. **Sleep**: Memory consolidation (SWS) and learning (REM).
12. **Binding Problem**: Solution via gamma synchronization.
13. **Active Inference (Loop Closure)**: Interaction with the environment.
14. **Death**: Cascade of destruction (anoxia, excitotoxicity).
15. **Anesthesia**: Mechanism of propofol action via GABA receptors.
16. **Affective Dimension**: Neurobiology of emotions (Valence/Arousal).

*(See the `Theory` folder for the full list)*

---

## 📄 License

See the [LICENSE](LICENSE) file.

---
*Developed within the framework of the Artificial Evolution of Intellegence.*
