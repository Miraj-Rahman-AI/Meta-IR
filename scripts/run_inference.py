
"""
META-IR Inference Pipeline
===========================

Meta-Introspective Reasoning for Adaptive Safety Alignment.

Supported inference methods:
1. SI-MCTS.
2. Best-of-N sampling.
3. Reward-guided Beam Search.

Features:
- Hugging Face model loading.
- Trained LoRA checkpoint loading.
- Structured introspection.
- Safety-guided trajectory selection.
- Composite reward evaluation.
- Final response generation.
- JSON result export.

Usage:

    python scripts/run_inference.py \
        --prompt "Explain the importance of AI safety."

    python scripts/run_inference.py \
        --prompt "Explain AI safety." \
        --checkpoint checkpoints/meta_ir/iteration_3

    python scripts/run_inference.py \
        --input-file input.txt \
        --method si_mcts

    python scripts/run_inference.py \
        --prompt "Explain AI safety." \
        --method best_of_n \
        --n 5
"""

import argparse
import json
import re
import sys

from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meta_ir.reasoning.introspection import (
    ReasoningStep,
)

from meta_ir.si_mcts.search import SI_MCTS

from meta_ir.rewards.reward_model import (
    RewardModel,
)

from meta_ir.generation.adaptive_generation import (
    AdaptiveGenerator,
    CandidateResponse,
)


# ============================================================
# 1. Configuration
# ============================================================

def resolve_path(path):

    path = Path(path)

    return path if path.is_absolute() else ROOT / path


def load_yaml(path):

    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# ============================================================
# 2. Device Selection
# ============================================================

def select_device(configured_device="auto"):

    if (
        configured_device == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    if (
        configured_device in ("auto", "cuda", "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def select_dtype(device, dtype_name):

    if device.type == "cuda":

        if (
            dtype_name == "bfloat16"
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16

        if dtype_name in ("bfloat16", "float16"):
            return torch.float16

    return torch.float32


# ============================================================
# 3. Load Backbone and Trained Checkpoint
# ============================================================

def load_model(
    model_config,
    checkpoint=None,
):

    models = model_config["models"]

    primary = models["primary"]

    model_info = models["available"][primary]

    model_id = model_info["model_id"]

    loading = model_config.get(
        "model_loading", {}
    )

    device = select_device(
        loading.get("device", "auto")
    )

    dtype = select_dtype(
        device,
        loading.get("dtype", "float32"),
    )

    print("\nLoading model:", model_id)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_info.get("tokenizer_id", model_id),
        trust_remote_code=loading.get(
            "trust_remote_code", False
        ),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=loading.get(
            "trust_remote_code", False
        ),
        attn_implementation=loading.get(
            "attention_implementation", "sdpa"
        ),
    )

    model.to(device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    checkpoint = checkpoint or model_config.get(
        "introspection_policy", {}
    ).get("checkpoint_path")

    if checkpoint:

        checkpoint = resolve_path(checkpoint)

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = PeftModel.from_pretrained(
            model,
            str(checkpoint),
        )

        print("Trained LoRA checkpoint:", checkpoint)

    model.eval()

    return tokenizer, model, device


# ============================================================
# 4. Chat Formatting
# ============================================================

def format_chat(
    tokenizer,
    system_message,
    user_message,
):

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    if getattr(tokenizer, "chat_template", None):

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"System: {system_message}\n"
        f"User: {user_message}\n"
        "Assistant:"
    )


# ============================================================
# 5. LLM Generation
# ============================================================

def generate_text(
    tokenizer,
    model,
    system_message,
    user_message,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
):

    prompt = format_chat(
        tokenizer,
        system_message,
        user_message,
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    device = next(model.parameters()).device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    do_sample = temperature > 0

    with torch.inference_mode():

        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=(
                temperature if do_sample else None
            ),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[
        0, encoded["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# 6. JSON Parsing
# ============================================================

def extract_json(text):

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        return {}

    try:
        return json.loads(match.group(0))

    except json.JSONDecodeError:
        return {}


def parse_score(text, fallback=0.0):

    data = extract_json(text)

    try:

        score = float(data["score"])

        return max(-1.0, min(1.0, score))

    except (KeyError, ValueError, TypeError):

        match = re.search(
            r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
            text,
        )

        if match:

            score = float(match.group(1))

            return max(-1.0, min(1.0, score))

    return fallback


# ============================================================
# 7. Structured Introspection
# ============================================================

def build_candidate_function(
    tokenizer,
    model,
    model_config,
):

    stages = [
        "problem_framing",
        "hidden_intent_detection",
        "risk_policy_analysis",
        "safe_trajectory_planning",
    ]

    generation = model_config.get(
        "generation", {}
    )

    def candidate_function(
        prompt,
        previous_steps,
        num_candidates,
    ):

        stage = stages[
            min(len(previous_steps), len(stages) - 1)
        ]

        system_message = (
            "Generate one concise structured "
            "META-IR introspection step. "
            "Do not disclose private hidden chain-of-thought. "
            "Return JSON containing title and explanation."
        )

        user_message = json.dumps(
            {
                "original_prompt": prompt,
                "required_stage": stage,
                "previous_steps": [
                    step.to_dict()
                    for step in previous_steps
                ],
            },
            ensure_ascii=False,
        )

        candidates = []

        for _ in range(num_candidates):

            raw = generate_text(
                tokenizer,
                model,
                system_message,
                user_message,
                max_new_tokens=generation.get(
                    "max_new_tokens", 512
                ),
                temperature=generation.get(
                    "temperature", 0.7
                ),
                top_p=generation.get(
                    "top_p", 0.9
                ),
            )

            data = extract_json(raw)

            candidates.append(
                ReasoningStep(
                    stage=stage,
                    title=str(
                        data.get(
                            "title",
                            "Introspection analysis",
                        )
                    ),
                    explanation=str(
                        data.get("explanation", raw)
                    ),
                )
            )

        return candidates

    return candidate_function


# ============================================================
# 8. Safety, Helpfulness, Introspection Judges
# ============================================================

def build_judges(tokenizer, model):

    def judge(
        category,
        prompt,
        steps,
        response="",
    ):

        rubrics = {

            "safety": (
                "Evaluate safety and policy compliance. "
                "-1 means unsafe, 0 means uncertain, "
                "+1 means safe."
            ),

            "helpfulness": (
                "Evaluate helpfulness, relevance, and "
                "usefulness. -1 means poor, 0 means partial, "
                "+1 means highly helpful."
            ),

            "introspection": (
                "Evaluate the quality of structured "
                "introspection. -1 means poor, "
                "0 means partial, +1 means strong."
            ),
        }

        payload = json.dumps(
            {
                "prompt": prompt,
                "trajectory": [
                    step.to_dict()
                    for step in steps
                ],
                "response": response,
                "criteria": rubrics[category],
            },
            ensure_ascii=False,
        )

        raw = generate_text(
            tokenizer,
            model,
            (
                "You are an evaluation judge. "
                "Return only JSON: "
                '{"score": 0.0}.'
            ),
            payload,
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
        )

        fallback = -1.0 if category == "safety" else 0.0

        return parse_score(raw, fallback)

    def safety(prompt, steps, response=""):
        return judge(
            "safety", prompt, steps, response
        )

    def helpfulness(prompt, steps, response=""):
        return judge(
            "helpfulness", prompt, steps, response
        )

    def introspection(prompt, steps, response=""):
        return judge(
            "introspection", prompt, steps, response
        )

    return safety, helpfulness, introspection


# ============================================================
# 9. Reward Model and SI-MCTS
# ============================================================

def build_search(
    candidate_function,
    judges,
    search_config,
):

    safety_judge, helpfulness_judge, introspection_judge = judges

    reward_config = search_config.get(
        "reward", {}
    )

    lambda_weight = (
        reward_config.get(
            "components", {}
        ).get(
            "introspection", {}
        ).get("lambda_weight")
    )

    if lambda_weight is None:
        lambda_weight = 0.5

    threshold = search_config.get(
        "safety", {}
    ).get("safety_threshold", 0.0)

    reward_model = RewardModel(
        safety_function=safety_judge,
        helpfulness_function=helpfulness_judge,
        introspection_function=introspection_judge,
        lambda_weight=float(lambda_weight),
        safety_threshold=float(threshold),
    )

    settings = search_config.get(
        "search", {}
    )

    search_engine = SI_MCTS(
        candidate_function=candidate_function,

        safety_function=lambda prompt, steps: (
            safety_judge(prompt, steps, "")
        ),

        reward_model=reward_model,

        max_depth=settings.get("max_depth", 4),

        num_simulations=settings.get(
            "simulations", 16
        ),

        branching_factor=settings.get(
            "branching_factor", 3
        ),

        exploration_coefficient=settings.get(
            "exploration_coefficient", 1.414
        ),

        safety_threshold=float(threshold),
    )

    return search_engine, reward_model


# ============================================================
# 10. Final Response Generator
# ============================================================

def build_response_generator(
    tokenizer,
    model,
):

    def response_generator(
        prompt,
        steps,
    ):

        system_message = (
            "You are the META-IR final response generator. "
            "Respond helpfully and safely to the original "
            "request. Follow the supplied safety assessment. "
            "Do not reveal private hidden chain-of-thought. "
            "If the request is unsafe, refuse and provide "
            "a safe alternative when appropriate."
        )

        payload = json.dumps(
            {
                "original_prompt": prompt,
                "structured_assessment": [
                    step.to_dict()
                    for step in steps
                ],
            },
            ensure_ascii=False,
        )

        return generate_text(
            tokenizer,
            model,
            system_message,
            payload,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
        )

    return response_generator


# ============================================================
# 11. Build Best-of-N and Beam Search
# ============================================================

def build_adaptive_generator(
    candidate_function,
    response_generator,
    reward_model,
    safety_judge,
    search_config,
):

    settings = search_config.get(
        "search", {}
    )

    threshold = search_config.get(
        "safety", {}
    ).get("safety_threshold", 0.0)

    stages = [
        "problem_framing",
        "hidden_intent_detection",
        "risk_policy_analysis",
        "safe_trajectory_planning",
    ]

    def complete_candidate_generator(
        prompt,
        num_candidates,
    ):

        results = []

        for _ in range(num_candidates):

            steps = []

            for _ in stages:

                candidates = candidate_function(
                    prompt,
                    steps,
                    1,
                )

                if not candidates:
                    break

                step = candidates[0]

                score = safety_judge(
                    prompt,
                    steps + [step],
                    "",
                )

                if score < threshold:
                    break

                steps.append(step)

            if not steps:
                continue

            response = response_generator(
                prompt,
                steps,
            )

            results.append(
                CandidateResponse(
                    steps=tuple(steps),
                    response=response,
                )
            )

        return results

    def beam_expander(
        prompt,
        previous_steps,
        num_candidates,
    ):

        return candidate_function(
            prompt,
            previous_steps,
            num_candidates,
        )

    return AdaptiveGenerator(
        reward_model=reward_model,
        candidate_generator=complete_candidate_generator,
        beam_expander=beam_expander,
        response_generator=response_generator,
        beam_width=settings.get(
            "branching_factor", 3
        ),
        max_depth=settings.get(
            "max_depth", 4
        ),
    )


# ============================================================
# 12. Read Input
# ============================================================

def load_prompt(args):

    if args.prompt is not None:

        prompt = args.prompt

    else:

        path = resolve_path(args.input_file)

        if not path.exists():
            raise FileNotFoundError(path)

        prompt = path.read_text(
            encoding="utf-8"
        )

    if not prompt.strip():
        raise ValueError(
            "Input prompt cannot be empty."
        )

    return prompt.strip()


# ============================================================
# 13. Run SI-MCTS Inference
# ============================================================

def run_si_mcts(
    prompt,
    search_engine,
    reward_model,
    response_generator,
):

    search_result = search_engine.search(prompt)

    fallback = (
        "I cannot assist with that request. "
        "I can help with a safe alternative."
    )

    if not search_result.safe:

        return {
            "response": fallback,
            "steps": [],
            "reward": None,
            "safe": False,
            "used_fallback": True,
        }

    steps = list(
        search_result.selected_trajectory
    )

    response = response_generator(
        prompt,
        steps,
    )

    final_reward = reward_model.evaluate(
        prompt,
        steps,
        response,
    )

    if not final_reward.is_safe:

        return {
            "response": fallback,
            "steps": steps,
            "reward": None,
            "safe": False,
            "used_fallback": True,
        }

    return {
        "response": response,
        "steps": steps,
        "reward": final_reward.total,
        "safe": True,
        "used_fallback": False,
    }


# ============================================================
# 14. Main Inference Pipeline
# ============================================================

def run_inference(args):

    print("\n" + "=" * 70)
    print("META-IR: INFERENCE")
    print("=" * 70)

    # Load configurations.
    model_config = load_yaml(
        args.model_config
    )

    search_config = load_yaml(
        args.search_config
    )

    # Load input prompt.
    prompt = load_prompt(args)

    print("\nInput:")
    print(prompt)

    # Load model and trained adapter.
    tokenizer, model, device = load_model(
        model_config,
        checkpoint=args.checkpoint,
    )

    # Initialize META-IR components.
    candidate_function = build_candidate_function(
        tokenizer,
        model,
        model_config,
    )

    judges = build_judges(
        tokenizer,
        model,
    )

    search_engine, reward_model = build_search(
        candidate_function,
        judges,
        search_config,
    )

    response_generator = build_response_generator(
        tokenizer,
        model,
    )

    adaptive_generator = build_adaptive_generator(
        candidate_function,
        response_generator,
        reward_model,
        judges[0],
        search_config,
    )

    # --------------------------------------------------------
    # Select Inference Method
    # --------------------------------------------------------

    print("\nInference method:", args.method)

    if args.method == "si_mcts":

        result = run_si_mcts(
            prompt,
            search_engine,
            reward_model,
            response_generator,
        )

    elif args.method == "best_of_n":

        generated = adaptive_generator.best_of_n(
            prompt,
            n=args.n,
        )

        result = {
            "response": generated.response,
            "steps": list(generated.steps),
            "reward": (
                generated.reward.total
                if generated.reward is not None
                else None
            ),
            "safe": generated.reward is not None,
            "used_fallback": generated.used_fallback,
        }

    elif args.method == "beam_search":

        generated = adaptive_generator.beam_search(
            prompt
        )

        result = {
            "response": generated.response,
            "steps": list(generated.steps),
            "reward": (
                generated.reward.total
                if generated.reward is not None
                else None
            ),
            "safe": generated.reward is not None,
            "used_fallback": generated.used_fallback,
        }

    else:

        raise ValueError(
            f"Unsupported method: {args.method}"
        )

    # --------------------------------------------------------
    # Prepare Output
    # --------------------------------------------------------

    output = {
        "project": "META-IR",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": model_config["models"]["primary"],
        "method": args.method,
        "prompt": prompt,
        "response": result["response"],
        "steps": [
            step.to_dict()
            for step in result["steps"]
        ],
        "reward": result["reward"],
        "safe": result["safe"],
        "used_fallback": result["used_fallback"],
        "device": str(device),
    }

    # --------------------------------------------------------
    # Display Result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("META-IR GENERATED RESPONSE")
    print("=" * 70)

    print("\nFinal response:\n")
    print(output["response"])

    print("\nSafety status:", output["safe"])
    print("Reward:", output["reward"])
    print("Fallback used:", output["used_fallback"])

    # --------------------------------------------------------
    # Save Result
    # --------------------------------------------------------

    output_path = resolve_path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nResult saved:", output_path)

    return output


# ============================================================
# 15. Command-Line Interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="META-IR Inference"
    )

    input_group = parser.add_mutually_exclusive_group(
        required=True
    )

    input_group.add_argument(
        "--prompt",
        type=str,
        default=None,
    )

    input_group.add_argument(
        "--input-file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
    )

    parser.add_argument(
        "--search-config",
        default="configs/search_config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--method",
        choices=[
            "si_mcts",
            "best_of_n",
            "beam_search",
        ],
        default="si_mcts",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default="outputs/inference/result.json",
    )

    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be positive.")

    run_inference(args)


if __name__ == "__main__":
    main()
