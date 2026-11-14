
"""
META-IR Jailbreak Task Simulation.

Organizes adversarial evaluation examples into task families.

This module loads existing benchmark examples. It does not
automatically generate harmful instructions or attacks.
"""

import json
import random

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class JailbreakTask:

    task_id: str

    family: str

    prompt: str

    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationRecord:

    task_id: str

    family: str

    prompt: str

    response: str


class JailbreakSimulator:
    """Benchmark task loader and sampling controller."""

    def __init__(
        self,
        tasks: Sequence[JailbreakTask],
        seed: int = 42,
    ):

        if not tasks:
            raise ValueError("Task collection cannot be empty.")

        self.tasks = list(tasks)

        self.rng = random.Random(seed)

        self.families = sorted({
            task.family for task in self.tasks
        })

    @classmethod
    def from_jsonl(
        cls,
        file_path: str,
        seed: int = 42,
    ) -> "JailbreakSimulator":

        tasks = []

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                record = json.loads(line)

                required = ("prompt", "family")

                if not all(key in record for key in required):
                    raise ValueError(
                        f"Missing fields at line {line_number}."
                    )

                tasks.append(
                    JailbreakTask(
                        task_id=str(
                            record.get("id", line_number)
                        ),
                        family=record["family"],
                        prompt=record["prompt"],
                        metadata=record.get("metadata", {}),
                    )
                )

        return cls(tasks=tasks, seed=seed)

    def sample_batch(
        self,
        batch_size: int,
    ) -> list[JailbreakTask]:

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        if batch_size <= len(self.tasks):
            return self.rng.sample(self.tasks, batch_size)

        return self.rng.choices(
            self.tasks,
            k=batch_size,
        )

    def sample_family(
        self,
        family: str,
        batch_size: int = 1,
    ) -> list[JailbreakTask]:

        candidates = [
            task for task in self.tasks
            if task.family == family
        ]

        if not candidates:
            raise ValueError(
                f"Unknown or empty task family: {family}"
            )

        return self.rng.choices(
            candidates,
            k=batch_size,
        )

    def run(
        self,
        agent: Callable[[str], str],
        tasks: Sequence[JailbreakTask],
    ) -> list[SimulationRecord]:
        """Run a supplied agent on selected benchmark tasks."""

        results = []

        for task in tasks:

            response = agent(task.prompt)

            results.append(
                SimulationRecord(
                    task_id=task.task_id,
                    family=task.family,
                    prompt=task.prompt,
                    response=str(response),
                )
            )

        return results
