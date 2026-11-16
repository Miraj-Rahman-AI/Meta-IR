
"""
META-IR Basic Inference Example
================================

Demonstrates:
1. Loading a Hugging Face causal language model.
2. Structured introspective reasoning.
3. Safety-Informed Monte Carlo Tree Search.
4. Composite reward evaluation.
5. Safe final response generation.

Usage:
    python examples/basic_inference.py \
        --prompt "Explain trustworthy AI."

    python examples/basic_inference.py \
        --prompt "Explain trustworthy AI." \
        --checkpoint checkpoints/meta_ir/iteration_3
"""

import argparse
import json
import re
import sys

from pathlib import Path

import torch
import yaml

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from meta_ir.reasoning.introspection import (
    ReasoningStep,
    StructuredIntrospector,
)

from meta_ir.si_mcts.search import SI_MCTS

from meta_ir.rewards.reward_model import (
    RewardModel,
)


# ============================================================
# 1. Configuration
# ============================================================

def load_yaml(path):

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# ============================================================
# 2. Model Initialization
# ============================================================

def load_model(model_config, checkpoint=None):

    models = model_config["models"]

    primary = models["primary"]

    model_info = models["available"][primary]

    model_id = model_info["model_id"]

    loading = model_config.get(
        "model_loading", {}
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print("Loading:", model_id)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_info.get("tokenizer_id", model_id),
        trust_remote_code=loading.get(
            "trust_remote_code", False
        ),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=loading.get(
            "trust_remote_code", False
        ),
    )

    model.to(device)

    if checkpoint:

        from peft import PeftModel

        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        model = PeftModel.from_pretrained(
            model,
            str(checkpoint_path),
        )

    model.eval()

    return tokenizer, model, device


# ============================================================
# 3. Text Generation
# ============================================================

def generate_text(
    tokenizer,
    model,
    system_message,
    user_message,
    max_new_tokens=512,
    temperature=0.2,
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

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    else:

        prompt = (
            f"System: {system_message}\n"
            f"User: {user_message}\n"
            "Assistant:"
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

    with torch.inference_mode():

        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=(
                temperature if temperature > 0 else None
            ),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[
        0, encoded["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# 4. JSON Parsing
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


def extract_score(text, fallback=-1.0):

    data = extract_json(text)

    try:

        score = float(data["score"])

        return max(-1.0, min(1.0, score))

    except (KeyError, ValueError, TypeError):
        return fallback


# ============================================================
# 5. Structured Introspection
# ============================================================

def build_candidate_function(tokenizer, model):

    stages = [
        "problem_framing",
        "hidden_intent_detection",
        "risk_policy_analysis",
        "safe_trajectory_planning",
    ]

    def candidate_function(
        prompt,
        previous_steps,
        num_candidates,
    ):

        stage = stages[
            min(len(previous_steps), len(stages) - 1)
        ]

        candidates = []

        for _ in range(num_candidates):

            raw = generate_text(
                tokenizer,
                model,
                (
                    "Generate a concise, structured "
                    "META-IR analysis step. Do not reveal "
                    "private hidden chain-of-thought. "
                    "Return JSON with title and explanation."
                ),
                json.dumps(
                    {
                        "prompt": prompt,
                        "stage": stage,
                        "previous_steps": [
                            step.to_dict()
                            for step in previous_steps
                        ],
                    },
                    ensure_ascii=False,
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
# 6. Evaluation Judges
# ============================================================

def build_judges(tokenizer, model):

    def judge(
        category,
        prompt,
        steps,
        response="",
    ):

        criteria = {
            "safety": (
                "Evaluate whether the response is safe. "
                "-1 = unsafe, 0 = uncertain, +1 = safe."
            ),
            "helpfulness": (
                "Evaluate usefulness and relevance. "
                "-1 = unhelpful, 0 = partial, +1 = helpful."
            ),
            "introspection": (
                "Evaluate structured introspection quality. "
                "-1 = poor, 0 = partial, +1 = strong."
            ),
        }

        raw = generate_text(
            tokenizer,
            model,
            (
                "You are an independent evaluation judge. "
                "Return only JSON in this format: "
                '{"score": 0.0}.'
            ),
            json.dumps(
                {
                    "prompt": prompt,
                    "steps": [
                        step.to_dict()
                        for step in steps
                    ],
                    "response": response,
                    "criteria": criteria[category],
                },
                ensure_ascii=False,
            ),
            max_new_tokens=64,
            temperature=0.0,
        )

        fallback = -1.0 if category == "safety" else 0.0

        return extract_score(raw, fallback)

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
# 7. Final Response Generation
# ============================================================

def generate_final_response(
    tokenizer,
    model,
    prompt,
    trajectory,
):

    return generate_text(
        tokenizer,
        model,
        (
            "You are the META-IR response generator. "
            "Provide a helpful, accurate, and safe response. "
            "Do not disclose private hidden chain-of-thought. "
            "Respect safety constraints."
        ),
        json.dumps(
            {
                "original_prompt": prompt,
                "structured_assessment": [
                    step.to_dict()
                    for step in trajectory
                ],
            },
            ensure_ascii=False,
        ),
        temperature=0.2,
    )


# ============================================================
# 8. Main META-IR Pipeline
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="META-IR Basic Inference"
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--output",
        default="outputs/examples/basic_inference.json",
    )

    args = parser.parse_args()

    model_config = load_yaml(
        "configs/model_config.yaml"
    )

    search_config = load_yaml(
        "configs/search_config.yaml"
    )

    tokenizer, model, device = load_model(
        model_config,
        args.checkpoint,
    )

    candidate_function = build_candidate_function(
        tokenizer,
        model,
    )

    introspector = StructuredIntrospector(
        candidate_function=candidate_function
    )

    safety, helpfulness, introspection = build_judges(
        tokenizer,
        model,
    )

    reward_settings = search_config.get(
        "reward", {}
    )

    lambda_weight = (
        reward_settings.get(
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
        safety_function=safety,
        helpfulness_function=helpfulness,
        introspection_function=introspection,
        lambda_weight=float(lambda_weight),
        safety_threshold=float(threshold),
    )

    settings = search_config.get(
        "search", {}
    )

    search_engine = SI_MCTS(
        candidate_function=introspector.generate_candidates,
        safety_function=lambda prompt, steps: (
            safety(prompt, steps, "")
        ),
        reward_model=reward_model,
        max_depth=settings.get("max_depth", 4),
        num_simulations=settings.get("simulations", 16),
        branching_factor=settings.get("branching_factor", 3),
        exploration_coefficient=settings.get(
            "exploration_coefficient", 1.414
        ),
        safety_threshold=float(threshold),
    )

    print("\nRunning META-IR SI-MCTS...")

    search_result = search_engine.search(
        args.prompt
    )

    fallback = (
        "I cannot assist with that request. "
        "I can help with a safe alternative."
    )

    if not search_result.safe:

        response = fallback
        final_reward = None

    else:

        trajectory = list(
            search_result.selected_trajectory
        )

        response = generate_final_response(
            tokenizer,
            model,
            args.prompt,
            trajectory,
        )

        final_reward = reward_model.evaluate(
            args.prompt,
            trajectory,
            response,
        )

        if not final_reward.is_safe:
            response = fallback

    output = {
        "project": "META-IR",
        "prompt": args.prompt,
        "response": response,
        "trajectory": (
            [
                step.to_dict()
                for step in search_result.selected_trajectory
            ]
            if search_result.safe
            else []
        ),
        "safe": (
            bool(final_reward.is_safe)
            if final_reward is not None
            else False
        ),
        "reward": (
            final_reward.total
            if final_reward is not None
            else None
        ),
        "device": str(device),
    }

    print("\n" + "=" * 60)
    print("META-IR RESPONSE")
    print("=" * 60)
    print(response)

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = ROOT / output_path

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


if __name__ == "__main__":
    main()
