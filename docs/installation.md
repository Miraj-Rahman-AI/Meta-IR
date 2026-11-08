
# META-IR Installation Guide

## 1. Overview

This document describes how to install and configure the META-IR research implementation.

META-IR is a Python-based research framework for adaptive safety alignment in Large Language Models (LLMs).

The installation process covers:

1. Python environment preparation.
2. Dependency installation.
3. GPU and CUDA verification.
4. Pretrained model preparation.
5. Dataset preparation.
6. Experiment configuration.
7. Installation verification.

---

## 2. System Requirements

### 2.1 Software Requirements

| Component | Requirement |
|---|---|
| Operating System | Linux recommended |
| Python | 3.10 or higher |
| Package Manager | pip |
| Deep Learning Framework | PyTorch |
| Transformer Framework | Hugging Face Transformers |
| Distributed Training | PyTorch Distributed / Accelerate |
| Configuration | YAML |
| Version Control | Git |

### 2.2 Hardware Requirements

The reported full-scale experiments use:

| Component | Experimental Configuration |
|---|---|
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| GPU Memory | 80 GB per GPU |
| CUDA | 12.2 |
| NVIDIA Driver | 535.129.03 |
| Numerical Precision | bfloat16 |
| Distributed Execution | Distributed Data Parallel |

These specifications describe the reported experimental environment, not the minimum requirements for every use case.

Smaller-scale inference and development may be possible with fewer GPUs, subject to model size, sequence length, and search requirements.

---

## 3. Clone the Repository

Clone the official META-IR repository:

```bash
git clone https://github.com/Miraj-Rahman-AI/Meta-IR.git

cd Meta-IR
```

---

## 4. Create a Python Environment

We recommend using an isolated virtual environment.

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

Upgrade the package installer:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 5. Install Dependencies

META-IR declares its runtime dependencies in `pyproject.toml` and provides a `requirements.txt` installation entry point.

Install the project and its runtime dependencies:

```bash
pip install -r requirements.txt
```

For development and testing, install the optional development dependencies:

```bash
pip install -e ".[dev]"
```

The development environment includes:

- pytest for automated testing.
- Ruff for Python code quality and linting.

### Main Dependencies

| Package | Purpose |
|---|---|
| PyTorch | Neural network computation and optimization |
| Transformers | Pretrained language models and tokenizers |
| Accelerate | Distributed and accelerated model execution |
| Datasets | Dataset loading and processing |
| NumPy | Numerical computation |
| PyYAML | Experiment configuration |
| Safetensors | Model-weight serialization |
| tqdm | Progress monitoring |
| pytest | Testing |
| Ruff | Code quality |

---

## 6. GPU and CUDA Configuration

META-IR uses PyTorch for model execution and training.

For GPU-based experiments, install a PyTorch build compatible with your operating system, NVIDIA driver, and CUDA environment.

Consult the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Verify the PyTorch installation:

```bash
python -c "import torch; print(torch.__version__)"
```

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check the detected GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

### Mixed Precision

The reported META-IR experiments use bfloat16 mixed precision.

The execution environment must support the selected numerical precision.

---

## 7. Pretrained Model Preparation

META-IR uses instruction-tuned LLM backbones.

The reported experimental models are:

| Model | Experimental Role |
|---|---|
| LLaMA-3.1-8B-Instruct | Main backbone evaluation |
| Qwen-2-7B-Instruct | Cross-backbone evaluation |

**Model identifier verification:** The supplementary material uses the name `Qwen-2.7B-Instruct`, whereas the main paper uses `Qwen-2-7B-Instruct`. Verify the exact checkpoint used in the experiments before configuring model downloads.

### Model Preparation Steps

1. Obtain the pretrained model from its authorized distribution source.
2. Review and accept the model's applicable license.
3. Configure model access and authentication where required.
4. Download the model weights and tokenizer.
5. Set the correct model identifier or local checkpoint path in the experiment configuration.

Model weights are not distributed as part of this source-code repository.

Do not commit model weights, access tokens, or private credentials to GitHub.

---

## 8. Dataset Preparation

META-IR uses multiple safety-related datasets.

| Dataset | Research Purpose |
|---|---|
| AdvBench | Harmful instruction evaluation |
| HarmBench | Safety and jailbreak evaluation |
| WildJailbreak | Benign and adversarial prompt coverage |
| DIR | Structured introspection warm-up |

### Dataset Preparation

1. Obtain the datasets from their authorized sources.
2. Review their individual licenses and usage requirements.
3. Apply the preprocessing procedures described in the research protocol.
4. Construct the unified seed corpus.
5. Prepare the structured introspection dataset.
6. Separate meta-training and meta-evaluation data.
7. Configure dataset paths in the experiment configuration.

The datasets are not included in this repository.

Users must obtain the original datasets independently and comply with their respective terms.

---

## 9. Experiment Configuration

META-IR separates model, training, search, and evaluation configurations.

Recommended configuration organization:

```text
configs/
│
├── model_config.yaml
├── training_config.yaml
├── search_config.yaml
└── evaluation_config.yaml
```

### Model Configuration

Specifies:

- Backbone model identifier.
- Tokenizer.
- Model checkpoint.
- Numerical precision.
- Generation settings.

### Training Configuration

Specifies:

- Inner-loop learning rate.
- Outer-loop meta-learning rate.
- Optimizer.
- Learning-rate scheduler.
- Batch configuration.
- Meta-learning settings.

### Search Configuration

Specifies:

- SI-MCTS search parameters.
- Candidate expansion.
- Search depth.
- Safety evaluation.
- Reward configuration.
- Pruning constraints.

### Evaluation Configuration

Specifies:

- Dataset locations.
- Evaluation splits.
- Random seeds.
- Evaluation metrics.
- Result output locations.

All numerical experiment settings should be taken from the actual experiment configurations rather than guessed.

---

## 10. Verify the Installation

Verify the principal dependencies:

```bash
python -c "
import torch
import transformers
import accelerate
import datasets
import numpy
import yaml

print('META-IR dependencies imported successfully.')
"
```

Verify the local package:

```bash
python -c "import meta_ir; print('META-IR package imported successfully.')"
```

Run automated tests:

```bash
pytest -q
```

Check code quality:

```bash
ruff check src scripts tests
```

The package import and test commands require the corresponding implementation modules and tests to be present.

---

## 11. Running META-IR

The following are the intended command interfaces for the repository. Keep them synchronized with the actual scripts and argument names in the released implementation.

### Training

```bash
python scripts/train.py \
    --config configs/training_config.yaml
```

### SI-MCTS Execution

```bash
python scripts/run_si_mcts.py \
    --config configs/search_config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
    --config configs/evaluation_config.yaml
```

### Inference

```bash
python scripts/run_inference.py \
    --config configs/model_config.yaml
```

### Distributed Training

For a training script that supports PyTorch Distributed Data Parallel:

```bash
torchrun \
    --standalone \
    --nproc_per_node=8 \
    scripts/train.py \
    --config configs/training_config.yaml
```

The distributed command is intended for the reported eight-GPU configuration and requires the corresponding distributed training implementation.

---

## 12. Troubleshooting

### CUDA Is Not Available

Verify the NVIDIA driver, CUDA-compatible PyTorch installation, and GPU visibility.

### CUDA Out of Memory

Consider reducing:

- Training batch size.
- Sequence length.
- Number of candidate trajectories.
- SI-MCTS search budget.

### Model Access Failure

Check model permissions, authentication, and checkpoint identifiers.

### Dataset Loading Failure

Verify dataset paths, file formats, and preprocessing requirements.

### Package Import Failure

Verify that:

- The `src/meta_ir/` package exists.
- `__init__.py` files are present where required.
- The project has been installed successfully.
- The active virtual environment is correct.

### Distributed Training Failure

Check GPU visibility, CUDA compatibility, process configuration, and distributed communication settings.

---

## 13. Installation Notes

The installation instructions describe the intended software environment for META-IR.

Actual training and evaluation requirements depend on the selected backbone, dataset size, search configuration, and available hardware.

For exact experimental reproduction, refer to:

[`reproduction.md`](reproduction.md)

---

## 14. References

- META-IR: Meta-Introspective Reasoning for Adaptive Safety Alignment in LLM.
- META-IR Supplementary Materials.
- PyTorch: https://pytorch.org/
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Hugging Face Datasets: https://huggingface.co/docs/datasets
