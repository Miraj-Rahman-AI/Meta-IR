
# META-IR: Meta-Introspective Reasoning for Adaptive Safety Alignment in LLM

<p align="center">

![Python](https://img.shields.io/badge/Python-Research%20Implementation-blue?logo=python)
![AI](https://img.shields.io/badge/Domain-LLM%20Safety-purple)
![Research](https://img.shields.io/badge/Research-Adaptive%20Safety-red)
![Conference](https://img.shields.io/badge/IEEE-ICME%202026-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Research%20Code-blue)

</p>

<p align="center">
  <b>Official Research Code for META-IR</b>
  <br>
  Meta-Introspective Reasoning for Adaptive Safety Alignment in Large Language Models
</p>



> [!CAUTION]
> $\color{red}{\textsf{\textbf{⚠ Warning: Harmful or offensive content may be present.}}}$


## 1. Overview

**META-IR** is a meta-introspective reasoning framework designed to improve the safety alignment, reasoning stability, and adaptive robustness of Large Language Models (LLMs) against evolving jailbreak attacks.

Existing safety alignment approaches often rely on relatively static safety policies and fixed evaluation mechanisms. Consequently, aligned language models may remain vulnerable to adversarial distribution shifts, role-play manipulation, forced-prefix attacks, and disguised harmful intentions.

META-IR addresses these limitations through a dual-loop architecture that integrates structured introspective reasoning, Safety-Informed Monte Carlo Tree Search (SI-MCTS), composite reward optimization, and meta-learning across diverse jailbreak task families.

Rather than relying exclusively on static refusal behavior, META-IR enables models to examine potential risks, refine intermediate reasoning trajectories, and adapt their introspection policies to previously unseen adversarial patterns.

This repository provides the research-oriented source code and implementation resources for the META-IR framework.

### Publication

**2026 IEEE International Conference on Multimedia & Expo (ICME 2026)**

- Conference: IEEE ICME 2026
- Location: Bangkok, Thailand
- Paper: META-IR: Meta-Introspective Reasoning for Adaptive Safety Alignment in LLM
- Publication Status: Accepted


## 2. Research Contributions

META-IR introduces four principal research contributions:

### 1 Structured Introspective Reasoning

A structured reasoning mechanism that enables LLMs to progressively assess user intent, identify potential safety risks, evaluate policy conflicts, and produce policy-consistent responses.

### 2 Safety-Informed Monte Carlo Tree Search (SI-MCTS)

A safety-aware search mechanism that explores alternative introspective reasoning trajectories, evaluates candidate paths, and prunes unsafe or policy-violating branches.

### 3 Dynamic Jailbreak Simulation

An adversarial simulation mechanism that exposes the framework to progressively challenging jailbreak prompts and diverse attack strategies.

### 4 Dual-Loop Meta-Learning

A meta-learning mechanism that adapts the introspection policy across different jailbreak task families to improve generalization to previously unseen adversarial patterns.


## 3. META-IR Architecture

META-IR integrates structured introspection, safety-informed search, reward optimization, and meta-learning into a unified adaptive safety alignment framework.

The architecture consists of the following major components:

### Component 1: Structured Introspective Reasoning

The model generates a structured introspective trajectory:

$$
y_{\mathrm{IR}} = (s_1, s_2, \ldots, s_n, f)
$$

where:

- $s_i$: An individual introspective reasoning step.
- $n$: Number of reasoning steps.
- $f$: Final model response.

The structured representation supports progressive risk assessment and evaluation of reasoning quality.

### Component 2: Safety-Informed Monte Carlo Tree Search (SI-MCTS)

SI-MCTS explores alternative introspective reasoning trajectories and evaluates their safety and usefulness.

Candidate trajectories are evaluated using a composite reward:

$$
R(H,S,I) = F(S)H + S + \lambda I
$$

where:

- $H$: Helpfulness.
- $S$: Safety.
- $I$: Introspection quality.
- $F(S)$: Safety-dependent modulation function.
- $\lambda$: Introspection reward coefficient.

Unsafe or policy-violating reasoning branches are pruned during the search process.

### Component 3: Dynamic Jailbreak Simulation

The framework considers diverse adversarial task families, including:

- Role-play framing.
- Forced-prefix injection.
- Disguised harmful intent.
- Narrative camouflage.
- Constrained-output attacks.

These task families support adaptive evaluation and meta-learning.

### Component 4: Dual-Loop Meta-Learning

META-IR separates trajectory refinement from cross-task policy adaptation.

**Inner Loop:**

- Generates structured introspective trajectories.
- Performs safety-informed search.
- Evaluates helpfulness, safety, and introspection quality.
- Refines candidate reasoning trajectories.

**Outer Loop:**

- Samples different jailbreak task families.
- Aggregates task-level rewards.
- Updates the introspection policy.
- Improves generalization across adversarial distributions.

The meta-learning objective is:

$$
J(\theta_r) =
\mathbb{E}_{\tau \sim \mathcal{T}}
\left[
R(H_\tau, S_\tau, I_\tau)
\right]
$$

where:

- $\theta_r$: Introspection policy parameters.
- $\mathcal{T}$: Distribution of jailbreak task families.
- $\tau$: A sampled jailbreak task.
- $H_\tau$: Helpfulness score for task $\tau$.
- $S_\tau$: Safety score for task $\tau$.
- $I_\tau$: Introspection quality score for task $\tau$.

The dual-loop mechanism enables META-IR to refine introspective reasoning trajectories while adapting its introspection policy across diverse jailbreak task families.


## 4. Key Features

| Feature | Description |
|---|---|
| Structured Introspection | Progressive safety and intent analysis |
| SI-MCTS | Safety-aware reasoning trajectory search |
| Composite Reward | Joint helpfulness, safety, and introspection evaluation |
| Jailbreak Simulation | Evaluation across diverse adversarial strategies |
| Meta-Learning | Adaptation across jailbreak task families |
| Adaptive Safety Alignment | Improved robustness under adversarial distribution shifts |
| Evaluation Framework | Safety, helpfulness, introspection, and generalization evaluation |


## 5. Repository Structure

The repository is organized into modular components to support implementation, experimentation, evaluation, and future research extensions.

```text
Meta-IR/
│
├── README.md
├── LICENSE
├── NOTICE.md
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── search_config.yaml
│   └── evaluation_config.yaml
│
├── src/
│   └── meta_ir/
│       ├── __init__.py
│       │
│       ├── reasoning/
│       │   ├── __init__.py
│       │   └── introspection.py
│       │
│       ├── si_mcts/
│       │   ├── __init__.py
│       │   ├── search.py
│       │   ├── node.py
│       │   └── pruning.py
│       │
│       ├── rewards/
│       │   ├── __init__.py
│       │   └── reward_model.py
│       │
│       ├── jailbreak/
│       │   ├── __init__.py
│       │   └── simulation.py
│       │
│       ├── meta_learning/
│       │   ├── __init__.py
│       │   ├── inner_loop.py
│       │   └── outer_loop.py
│       │
│       ├── generation/
│       │   ├── __init__.py
│       │   └── adaptive_generation.py
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── safety.py
│           ├── helpfulness.py
│           └── introspection.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_inference.py
│
├── examples/
│   └── basic_inference.py
│
├── tests/
│   ├── test_reasoning.py
│   ├── test_si_mcts.py
│   └── test_rewards.py
│
├── docs/
│   ├── architecture.md
│   ├── installation.md
│   └── reproduction.md
│
└── .github/
    └── workflows/
        └── tests.yml
```


## 6. Models and Experimental Setup

The research evaluates META-IR using two LLM backbone families.

| Model | Purpose |
|---|---|
| LLaMA-3.1-8B-Instruct | Main backbone evaluation |
| Qwen-2-7B-Instruct | Cross-backbone evaluation |

The exact Qwen checkpoint identifier should be verified against the experimental configuration before release.

### Experimental Resources

The supplementary experimental setup reports the following configuration:

| Component | Configuration |
|---|---|
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| CUDA | 12.2 |
| Precision | bfloat16 |
| Distributed Training | Distributed Data Parallel |
| Gradient Synchronization | All-Reduce |
| Optimizer | AdamW |
| Learning Rate Scheduling | Cosine scheduling with linear warm-up |

Backbone parameters are kept frozen during the reported meta-training process, while the introspection policy and selected reward-related meta-parameters are optimized.


## 7. Datasets and Benchmarks

The experimental framework uses a unified seed corpus constructed from the following safety-related datasets:

| Dataset | Role |
|---|---|
| AdvBench | Harmful instruction evaluation |
| HarmBench | Safety and jailbreak evaluation |
| WildJailbreak | Benign and adversarial prompt coverage |
| DIR | Dedicated structured introspection dataset |

The reported experimental setup includes:

- A 50K-prompt unified seed corpus.
- A 20K-prompt Dedicated Introspection Dataset (DIR).
- 10K safety-oriented prompts.
- 10K helpfulness-oriented prompts.
- Separate meta-training and meta-evaluation task families.

**Dataset availability:** The original datasets are not included in this source-code repository. Users should obtain them from their respective official sources and comply with their individual licenses and usage requirements.

The DIR dataset should only be redistributed if its contents and underlying data licenses permit redistribution.


## 8. Installation

### Prerequisites

- Python version supported by the project configuration.
- PyTorch.
- CUDA-compatible GPU for accelerated experiments.
- Access to the required pretrained model checkpoints.
- Access to the required benchmark datasets.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Miraj-Rahman-AI/Meta-IR.git

cd Meta-IR
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### Step 4: Verify the Environment

```bash
python --version

python -c "import torch; print(torch.__version__)"

python -c "import torch; print(torch.cuda.is_available())"
```


## 9. Configuration

META-IR uses configuration files to organize model settings, training parameters, search behavior, and evaluation settings.

Example configuration organization:

```text
configs/
│
├── model_config.yaml
├── training_config.yaml
├── search_config.yaml
└── evaluation_config.yaml
```

The configuration files should specify the following information:

**Model Configuration**
- Backbone model identifier.
- Tokenizer.
- Precision.
- Maximum sequence length.
- Generation parameters.

**Training Configuration**
- Learning rates.
- Optimizer.
- Batch size.
- Number of training iterations.
- Meta-learning parameters.
- Gradient accumulation.

**SI-MCTS Configuration**
- Search depth.
- Candidate expansion.
- Reward calculation.
- Safety evaluation.
- Branch pruning.

**Evaluation Configuration**
- Dataset paths.
- Evaluation splits.
- Random seeds.
- Evaluation metrics.
- Output directories.


## 10. Training

META-IR follows a dual-loop optimization process.

### Stage 1: Structured Introspection Warm-Up

The model is exposed to structured introspection examples to establish the required reasoning format.

### Stage 2: Inner-Loop Refinement

SI-MCTS generates and evaluates candidate reasoning trajectories using the composite reward mechanism.

### Stage 3: Outer-Loop Meta-Training

The introspection policy is updated using aggregated feedback across different jailbreak task families.

### Stage 4: Adaptive Evaluation

The trained framework is evaluated against static and adaptive jailbreak scenarios.

Example training entry point:

```bash
python scripts/train.py \
    --config configs/training_config.yaml
```

The actual script arguments should follow the released implementation.


## 11. Inference

The inference component is intended to demonstrate how META-IR processes an input prompt through structured introspection and safety-aware response generation.

Example command:

```bash
python scripts/run_inference.py \
    --config configs/model_config.yaml
```

A demonstration should report:

1. Input prompt.
2. Structured introspective analysis.
3. Safety assessment.
4. Search and refinement information.
5. Final model response.

The implementation should expose only appropriate diagnostic information and should not require disclosure of private internal model reasoning.


## 12. Evaluation

META-IR is evaluated across multiple dimensions of safety alignment and reasoning reliability.

### Safety and Harmlessness

- Adversarial attack performance.
- Harmfulness-related evaluation.
- Safety Violation Rate (SVR).
- Refusal Stability under Perturbation (RST).

### Helpfulness and Generalization

- Benign instruction-following performance.
- Meta-generalization.
- General reasoning performance.

### Introspection Quality

- Introspection Accuracy (IA).
- Risk-Explanation Attribution Recall (REAR).
- Meta-Confusion Rate (MCR).
- Jailbreak-Introspection Failure Rate (JIR).

### Meta-Learning Performance

- Meta-Training Adaptation (MTA).
- Adaptation performance across different jailbreak task families.

Example evaluation entry point:

```bash
python scripts/evaluate.py \
    --config configs/evaluation_config.yaml
```

Evaluation results should be reported with the corresponding dataset splits, model checkpoints, random seeds, and metric definitions.


## 13. Reproducibility

Reproducibility is an important objective of this research repository.

The following information should accompany experimental releases:

- Model checkpoint identifiers.
- Dataset versions and preprocessing procedures.
- Training and evaluation configurations.
- Random seeds.
- Optimizer and scheduler settings.
- SI-MCTS search parameters.
- Meta-learning parameters.
- Evaluation scripts.
- Software and hardware configurations.

### Reproducibility Guidelines

1. Use the same model checkpoint as the corresponding experiment.
2. Maintain consistent dataset splits.
3. Record all random seeds.
4. Use the published configuration settings.
5. Report the hardware and software environment.
6. Evaluate all methods under consistent experimental conditions.

The reported eight-GPU experimental configuration is the reference setup, not a requirement for running every demonstration. Smaller configurations may require reduced batch sizes, search budgets, or sequence lengths and should not be assumed to reproduce the reported results exactly.


## 14. Research Applications

META-IR provides a research framework for investigating:

- Adaptive safety alignment in LLMs.
- Jailbreak robustness.
- Structured introspective reasoning.
- Safety-aware search algorithms.
- Meta-learning for adversarial generalization.
- Reasoning stability under distribution shifts.
- Safety and helpfulness optimization.
- Self-improving language model systems.

The modular architecture also provides opportunities for further research into adaptive reasoning and trustworthy AI systems.


## 15. Limitations and Responsible Use

META-IR is a research framework and should not be interpreted as a guarantee of complete protection against jailbreak attacks or harmful model behavior.

Researchers should consider:

- Limitations of the underlying backbone models.
- Potential weaknesses in safety and reward evaluation.
- Distribution shifts beyond the evaluated attack families.
- Computational requirements of SI-MCTS and meta-learning.
- Dataset coverage and benchmark limitations.
- Potential differences between reported and independently reproduced results.

The framework should be evaluated carefully before deployment in safety-critical applications.


## 16. License

This repository is intended to be released under the **MIT License**.

The MIT License applies to the original source code distributed in this repository, subject to the rights and permissions of the respective code contributors.

Third-party datasets, pretrained models, dependencies, and external resources remain subject to their own licenses and usage policies.

See the [LICENSE](LICENSE) file for details.


## 17. Citation

If you use META-IR in your research, please cite the corresponding conference paper.

```bibtex
@inproceedings{miraj2026metair,
  author = {
    Miraj, Mahabubur Rahman and
    Huang, Hongyu and
    Mu, Nankun and
    Chen, Chao and
    Yi, Shanqin
  },
  title = {
    META-IR: Meta-Introspective Reasoning for
    Adaptive Safety Alignment in LLM
  },
  booktitle = {
    2026 IEEE International Conference on
    Multimedia and Expo (ICME)
  },
  year = {2026},
  address = {Bangkok, Thailand}
}
```



## 18. Acknowledgements

This work was supported by the **National Natural Science Foundation of China** (Grant No. 62322601) and the **Fundamental Research Funds for the Central Universities** (Grant Nos. 2024IAIS-QN017 and 2025CDJZDG F001).

The authors gratefully acknowledge the financial support provided by these funding programs.


## 19. Contact

**Research Project:** META-IR

**GitHub Repository:** https://github.com/Miraj-Rahman-AI/Meta-IR

**Email:** [miraj.ai.edu.cn@gmail.com](mailto:miraj.ai.edu.cn@gmail.com)

For questions, implementation discussions, reproducibility issues, or research collaboration, please feel free to contact me via email or use the GitHub Issues section of this repository.

---



<div align="center">
  <strong>META-IR — Adaptive Safety Alignment through Meta-Introspective Reasoning</strong><br>
  <sub>Research Code | IEEE ICME 2026</sub>
</div>


