
# META-IR Reproduction Guide

## 1. Overview

This document describes the experimental methodology and reproduction protocol for META-IR.

META-IR investigates adaptive safety alignment in Large Language Models through:

- Structured introspective reasoning.
- Safety-Informed Monte Carlo Tree Search (SI-MCTS).
- Composite reward optimization.
- Dynamic jailbreak simulation.
- Dual-loop meta-learning.
- Meta-adaptive response generation.

The objective of this guide is to provide a consistent experimental procedure for researchers who want to reproduce, evaluate, or extend the META-IR framework.


## 2. Experimental Objectives

The experimental evaluation investigates five principal questions:

1. Can structured introspection improve the identification of harmful intent and policy conflicts?
2. Can SI-MCTS improve the selection of safe and useful reasoning trajectories?
3. Can safety-prioritized reward optimization reduce unsafe response generation?
4. Can meta-learning improve adaptation across different jailbreak task families?
5. Can META-IR improve safety without significantly degrading helpfulness and general reasoning performance?

The experiments evaluate both safety-related and utility-related performance.


## 3. Experimental Models

The research evaluates META-IR using two instruction-tuned language-model backbones.

| Model | Experimental Purpose |
|---|---|
| LLaMA-3.1-8B-Instruct | Main backbone evaluation |
| Qwen-2-7B-Instruct | Cross-backbone evaluation |

The two backbones provide a basis for investigating whether the proposed framework generalizes across different model architectures and instruction-tuning pipelines.

**Checkpoint naming note:** The main paper identifies the second model as `Qwen-2-7B-Instruct`, while the supplementary material identifies it as `Qwen-2.7B-Instruct`. This discrepancy must be resolved against the actual experimental checkpoint before claiming exact reproduction.

### Backbone Parameter Policy

The supplementary training protocol specifies that the backbone parameters remain frozen.

Only the introspection-policy parameters and a small set of reward-related meta-parameters are optimized.

This separation is intended to preserve the backbone's general language capabilities during meta-adaptation.


## 4. Datasets and Experimental Corpus

### 4.1 Unified Seed Corpus

The experimental framework uses a unified 50K-prompt seed corpus constructed from three safety-related benchmarks:

1. AdvBench.
2. HarmBench.
3. WildJailbreak.

The main paper describes approximately 2K harmful examples from AdvBench and HarmBench combined, with approximately 48K additional examples subsampled from WildJailbreak.

The corpus provides harmful, adversarial, and benign prompt coverage.

### 4.2 Structured Introspection Dataset (DIR)

META-IR additionally uses a dedicated 20K-prompt structured introspection dataset.

| Dataset Component | Number of Prompts |
|---|---:|
| Safety-oriented prompts | 10,000 |
| Helpfulness-oriented prompts | 10,000 |
| Total DIR | 20,000 |

The prompts are converted into the structured introspective reasoning format.

DIR is used for structured introspection warm-up and is kept disjoint from meta-learning and evaluation datasets.

### 4.3 Inner-Loop Training Samples

The supplementary material reports that each META-IR inner-loop update uses:

| Sample Type | Number |
|---|---:|
| Safety introspection samples | 5,000 |
| Helpfulness introspection samples | 5,000 |
| Total | 10,000 |

### 4.4 Adaptive Jailbreak Samples

The supplementary protocol introduces additional adversarial jailbreak prompts from the adversarial split of WildJailbreak.

The final two meta-adaptation iterations use:

- 1,000 adversarial prompts.
- 2,000 adversarial prompts.

The corresponding iteration schedule should be maintained when reproducing the reported experiment.

### 4.5 Dataset Separation

To prevent information leakage:

- Separate meta-training and meta-evaluation prompts.
- Maintain disjoint evaluation data.
- Separate task families at the family level where applicable.
- Do not use evaluation prompts for structured introspection warm-up.
- Record dataset versions and preprocessing procedures.

### 4.6 Dataset Availability

The original datasets are not included in this repository.

Researchers must obtain the datasets from their respective authorized sources and comply with their licenses and usage policies.


## 5. Jailbreak Task Families

META-IR models adversarial behavior using task families rather than treating every prompt as an independent attack.

The supplementary material identifies five principal task families.

| Task Family | Description |
|---|---|
| Role-Play Deception | Harmful intent embedded within fictional or hypothetical narratives. |
| Forced-Compliance Prefixes | Directives intended to force the model to bypass safety constraints. |
| Narrative Camouflage | Benign storytelling used to obscure an unsafe objective. |
| Instruction Stacking | Multiple instructions combined to gradually escalate toward unsafe behavior. |
| Output-Constrained Evasion | Formatting or response restrictions used to induce policy violations. |

Each task family represents a distribution of prompts sharing a common adversarial mechanism but differing in surface realization.

The meta-learning objective is to improve generalization across these families.


## 6. Training Protocol

META-IR follows a staged training and self-improvement process.

### Stage 1: Structured Introspection Warm-Up

Objective:

Establish a consistent structured introspective reasoning format.

Procedure:

1. Prepare the 20K DIR dataset.
2. Separate safety-oriented and helpfulness-oriented examples.
3. Generate or load structured introspection training examples.
4. Apply the structured reasoning template.
5. Perform the reported supervised warm-up stage.

The structured format includes:

- Problem framing.
- Hidden malicious intent detection.
- Risk and policy analysis.
- Safe trajectory planning.
- Final response generation.

### Stage 2: SI-MCTS Trajectory Refinement

Objective:

Improve introspective reasoning trajectories through safety-informed search.

Procedure:

1. Receive an input prompt.
2. Generate an initial structured introspective trajectory.
3. Represent partial trajectories as search-tree nodes.
4. Expand candidate reasoning steps.
5. Evaluate node safety.
6. Prune nodes with negative safety scores.
7. Complete safe rollouts.
8. Evaluate helpfulness, safety, and introspection quality.
9. Backpropagate rewards from safe rollouts.


The safety-pruning condition is:

$$
S(x,z_k)<0 \Rightarrow p_k=1
$$

where:

- $S(x,z_k)$: Safety score of node $z_k$.
- $p_k=1$: The node is pruned.
- $p_k=0$: The node remains eligible for exploration.

Equivalently:

$$
S(x,z_k)\geq 0 \Rightarrow p_k=0
$$

Unsafe completed rollouts do not contribute to ancestor-node value updates.


### Stage 3: Composite Reward Evaluation

Evaluate each completed trajectory using:

$$
R(H,S,I)=F(S)H+S+\lambda I
$$

where:

- H represents helpfulness.
- S represents safety compliance.
- I represents introspection quality.
- F(S) controls the contribution of helpfulness.
- Lambda controls the contribution of introspection quality.

The reward modulation function is:

$$
F(S)=
\begin{cases}
0,&S<0\\
1,&S\geq0
\end{cases}
$$

The safety, helpfulness, and introspection evaluators are kept fixed during training and inference according to the supplementary protocol.

### Stage 4: Outer-Loop Meta-Learning

Objective:

Improve the introspection policy across different jailbreak task families.

Procedure:

1. Sample a batch of jailbreak task families.
2. Sample prompts from each task.
3. Generate refined trajectories using SI-MCTS.
4. Calculate the composite reward for each trajectory.
5. Aggregate rewards at the task level.
6. Update the introspection policy.
7. Repeat across meta-iterations.

The task-level reward is:

$$
R_\tau =
\mathbb{E}_{x\sim\tau}
[R(H(x),S(x),I(x))]
$$

The meta-learning objective is:

$$
J(\theta_r)=
\mathbb{E}_{\tau\sim\mathcal{T}}[R_\tau]
$$

The outer-loop update is:

$$
\theta_r\leftarrow
\theta_r+
\alpha_{\mathrm{meta}}
\nabla_{\theta_r}J(\theta_r)
$$

The inner SI-MCTS search is treated as a trajectory-generation process without gradient propagation through the search tree. The outer loop performs the meta-update.


## 7. Optimization and Training Environment

The supplementary material reports the following optimization practices.

| Setting | Reported Configuration |
|---|---|
| Optimizer | AdamW |
| Inner-Loop Learning Rate | Separate parameter, alpha_in |
| Outer-Loop Learning Rate | Separate parameter, alpha_meta |
| Weight Decay | Applied to non-bias and non-LayerNorm parameters |
| Gradient Control | Gradient clipping |
| Learning-Rate Scheduler | Cosine scheduling |
| Warm-Up | Linear warm-up |
| Numerical Precision | bfloat16 |
| Distributed Execution | Distributed Data Parallel |
| Backbone Parameters | Frozen |
| Introspection Policy | Meta-optimized |

The inner and outer loops use distinct learning rates.

The supplementary material does not provide the numerical values of all learning rates, weight decay, gradient clipping threshold, or warm-up schedule. These values must be obtained from the actual experiment configurations.

### Reference Hardware

| Hardware / Software | Configuration |
|---|---|
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| CUDA | 12.2 |
| NVIDIA Driver | 535.129.03 |
| Precision | bfloat16 |
| Distributed Training | DDP |
| Gradient Synchronization | All-Reduce |

SI-MCTS simulations are parallelized across GPUs, with rewards aggregated before the corresponding meta-update.


## 8. Meta-Adaptive Response Generation

After meta-learning, META-IR performs test-time response generation using the updated introspection policy.

The reported generation pipeline consists of:

### 8.1 Process Reward Model

A process reward model evaluates candidate reasoning trajectories and their safety and usefulness.

The reported process reward model uses Bradley-Terry preference modeling.

### 8.2 Best-of-N Sampling

Generate multiple candidate continuations and evaluate them using the process reward model.

Select the highest-reward candidate.

### 8.3 Reward-Guided Beam Search

For more difficult cases:

- Expand promising reasoning trajectories.
- Evaluate partial or complete trajectories.
- Prune unsafe continuations.
- Retain high-reward candidates.

### 8.4 Final Response

Select the highest-reward safe candidate as the final response.

The numerical Best-of-N candidate count, beam width, and decoding settings must be taken from the actual experiment configuration.


## 9. Evaluation Protocol

META-IR is evaluated across three major dimensions:

1. Harmlessness and safety.
2. Helpfulness and generalization.
3. Introspection reliability and meta-learning.

### 9.1 Harmlessness and Safety Metrics

| Metric | Description | Desired Direction |
|---|---|---|
| Adv | Adversarial attack success rate. | Lower |
| Harm | Overall harmfulness rate. | Lower |
| WJ-H | Harmful response rate on WildJailbreak. | Lower |
| SVR | Safety violation rate. | Lower |
| RST | Refusal stability under perturbation. | Higher |

### 9.2 Helpfulness and Generalization Metrics

| Metric | Description | Desired Direction |
|---|---|---|
| WJ-B | Benign helpfulness rate on WildJailbreak. | Higher |
| MGS | Multi-style generalization score. | Higher |
| MTBench | Instruction-following quality. | Higher |
| Arena Elo | Pairwise preference performance. | Higher |
| BBH | Big-Bench Hard reasoning accuracy. | Higher |

### 9.3 Introspection Reliability and Meta-Learning Metrics

| Metric | Description | Desired Direction |
|---|---|---|
| IA | Introspection Accuracy. | Higher |
| REAR | Risk-Explanation Attribution Recall. | Higher |
| MCR | Meta-Confusion Rate. | Lower |
| JIR | Jailbreak-Introspection Failure Rate. | Lower |
| MTA | Meta-Training Adaptation. | Higher |

### Metric Interpretation

- **IA:** Measures whether the introspective phase correctly identifies unsafe or policy-violating intent.
- **REAR:** Measures whether the reasoning explanation correctly attributes the underlying safety risk.
- **MCR:** Measures contradictions or instability in introspective reasoning across interaction turns.
- **JIR:** Measures cases where introspection appears correct but the final response remains unsafe.
- **MTA:** Measures the ability to recover safety-aligned behavior after exposure to distribution-shifted or adaptive jailbreak tasks.


## 10. Baseline Comparisons

The main paper describes comparisons with the following baseline approaches:

| Baseline | Description |
|---|---|
| Base | Instruction-tuned model without introspection or self-generation. |
| CoT | Supervised chain-of-thought reasoning. |
| SFT | Supervised fine-tuning alignment. |
| RLHF | Reinforcement learning from human feedback. |
| DPO | Direct preference optimization. |
| SACPO | Safety-critic-based preference optimization. |
| Self-Rewarding (RLOO) | Self-generated reward-based optimization. |
| STAIR-SFT | Fixed-format introspective reasoning without meta-level adaptation. |

All baselines should be evaluated using consistent backbone models, dataset splits, and evaluation procedures.

## 11. Reproduction Commands

The following commands represent the intended repository execution interface.

### Step 1: Environment Preparation

```bash
git clone https://github.com/Miraj-Rahman-AI/Meta-IR.git

cd Meta-IR

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Prepare Datasets and Configurations

Verify the required dataset files and model checkpoints.

Review:

```text
configs/
    model_config.yaml
    training_config.yaml
    search_config.yaml
    evaluation_config.yaml
```

### Step 3: Structured Introspection Warm-Up

Use the training configuration to execute the structured introspection warm-up stage.

### Step 4: Meta-Training

```bash
python scripts/train.py \
    --config configs/training_config.yaml
```

### Step 5: SI-MCTS Experiment

```bash
python scripts/run_si_mcts.py \
    --config configs/search_config.yaml
```

### Step 6: Evaluation

```bash
python scripts/evaluate.py \
    --config configs/evaluation_config.yaml
```

### Step 7: Inference

```bash
python scripts/run_inference.py \
    --config configs/model_config.yaml
```

These commands must match the actual scripts and command-line arguments in the released implementation.


## 12. Experimental Reporting

For reproducible evaluation, record the following information for each experiment:

| Category | Information to Record |
|---|---|
| Model | Backbone identifier and checkpoint version |
| Dataset | Dataset source and version |
| Data Split | Training, meta-training, and evaluation splits |
| Training | Optimizer, learning rates, and training steps |
| SI-MCTS | Search parameters and pruning settings |
| Reward | Helpfulness, safety, and introspection scores |
| Meta-Learning | Task families and meta-iterations |
| Hardware | GPU model and number of GPUs |
| Software | Python, PyTorch, CUDA, and Transformers versions |
| Evaluation | Metrics, judge models, and evaluation settings |
| Randomness | Random seeds and sampling configuration |

Where repeated trials are conducted, report the number of runs and appropriate statistical summaries.


## 13. Reproduction Checklist

Before considering an experiment reproducible, verify:

- [ ] Exact pretrained model checkpoints are identified.
- [ ] Dataset versions are recorded.
- [ ] Dataset preprocessing is documented.
- [ ] DIR contains the intended 20K examples.
- [ ] Meta-training and evaluation splits are separated.
- [ ] SI-MCTS safety pruning is implemented.
- [ ] Safe-only reward propagation is implemented.
- [ ] Composite reward calculations are verified.
- [ ] Inner and outer learning rates are recorded.
- [ ] Optimizer and scheduler settings are recorded.
- [ ] Meta-learning task families are documented.
- [ ] Random seeds are recorded.
- [ ] Evaluation judge models are identified.
- [ ] All reported metrics have documented definitions.
- [ ] Training and evaluation commands are tested.


## 14. Reproduction Limitations and Missing Configuration Details

The paper and supplementary material provide the main methodology and reference experimental environment, but they do not specify every numerical hyperparameter required for an exact independent reproduction.

The following values must be obtained from the actual implementation or experiment configuration:

| Configuration | Required Information |
|---|---|
| Inner-loop learning rate | Exact alpha_in |
| Outer-loop learning rate | Exact alpha_meta |
| Introspection reward coefficient | Exact lambda |
| UCB selection | Exploration coefficient |
| SI-MCTS | Search depth, branching, and rollout budget |
| Candidate generation | Number of candidate trajectories |
| Process Reward Model | Exact checkpoint and training configuration |
| Best-of-N | Candidate count |
| Beam Search | Beam width and decoding settings |
| Optimization | Batch sizes, training steps, and warm-up ratio |
| Evaluation | Judge model identifiers and random seeds |
| Dataset | Exact split files and preprocessing hashes |

### Methodological clarification

The main paper mentions optional conversion of composite rewards into pairwise preferences for step-level DPO optimization.

The supplementary training protocol describes frozen backbone parameters and optimization of the introspection policy and reward-related meta-parameters.

This documentation follows the supplementary material's explicit dual-loop training protocol. Any optional DPO stage should be documented separately if it is part of the actual released implementation.

The Qwen checkpoint naming discrepancy between the main paper and supplementary material must also be resolved before claiming exact reproduction.


## 15. Responsible Research and Safety

META-IR is intended for research into trustworthy and adaptive LLM safety alignment.

Researchers should:

- Use benchmark datasets according to their licenses.
- Avoid redistributing restricted model weights or datasets.
- Evaluate both harmful and benign behavior.
- Report residual safety failures.
- Preserve the distinction between measured results and theoretical expectations.
- Avoid interpreting improved benchmark performance as a guarantee of complete safety.

The reported experiments show that META-IR reduces safety failures but does not eliminate them.

External safety enforcement and appropriate human oversight remain important for deployment.



## 17. Citation

If you use META-IR in your research, please cite our paper:

**arXiv Preprint:**

```bibtex
@misc{miraj2026metair,
  title = {{META-IR}: Meta-Introspective Reasoning for Adaptive Safety Alignment in {LLM}},
  author = {
    Miraj, Mahabubur Rahman and
    Huang, Hongyu and
    Mu, Nankun and
    Chen, Chao and
    Yi, Shanqin
  },
  year = {2026},
  eprint = {2611.10089},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2611.10089}
}
```

**arXiv:** https://arxiv.org/abs/2611.10089



## 17. Contact

**Research Project:** META-IR

**Email:** miraj.ai.edu.cn@gmail.com

**GitHub:** https://github.com/Miraj-Rahman-AI/Meta-IR

For questions regarding experimental reproduction, implementation details, or research collaboration, please contact the project author or open a GitHub issue.
