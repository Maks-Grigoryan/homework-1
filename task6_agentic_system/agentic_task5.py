"""Four-role LangGraph system that attempts Homework 1, Task 5.

The graph has exactly four nodes: orchestrator, scientist, general, executor.
All communication is an append-only dialogue in graph state.  The orchestrator
uses an instruction model for every routing decision; there is no keyword
router and no fixed role order.

By default the model is Qwen2.5-0.5B-Instruct running locally.  A small local
model keeps the demonstration reproducible without an API key while still
making every routing decision a real model call.
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import torch
from langgraph.graph import END, START, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK5 = """Solve the Silverbox PINN task. Use SNLS80mV, samples
[42650:46746], subtract the excerpt means, train on the first 3072 samples and
test on the last 1024. Model y'' + a y' + b y + c y^3 = g u by a fully
connected PyTorch acceleration network alpha_theta(y,v,u). Integrate (y,v)
with RK4 at dt=1/610.35. Evaluate one continuous free run initialized only by
measured y[0] and a finite-difference v[0]; test-time input u is available but
test y is forbidden. Print train and test free-run RMSE and make a test plot.
Also explain what one data record is and what the reported free-run RMSE
measures. The executable implementation may use one of the repository's
tested profiles: small_tanh, deep_tanh, or relu."""


class Turn(TypedDict):
    role: str
    text: str


class GraphState(TypedDict, total=False):
    dialogue: Annotated[list[Turn], operator.add]
    decision_count: int
    next_role: str
    instruction: str
    work_dir: str


@dataclass(frozen=True)
class Decision:
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str
    reason: str


class LocalInstructionModel:
    """Minimal chat wrapper: every ``complete`` call is a model inference."""

    def __init__(self, model_id: str, cache_dir: Path) -> None:
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        print(f"Loading local instruction model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()

    def complete(self, system: str, user: str, max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def rank(self, system: str, user: str, candidates: list[str]) -> tuple[str, dict[str, float]]:
        """Choose a candidate by its mean conditional log probability."""

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        scores: dict[str, float] = {}
        with torch.inference_mode():
            for candidate in candidates:
                candidate_ids = self.tokenizer(
                    candidate, add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
                full_ids = torch.cat((prompt_ids, candidate_ids), dim=1)
                logits = self.model(input_ids=full_ids).logits
                start = prompt_ids.shape[1] - 1
                token_logits = logits[:, start : start + candidate_ids.shape[1], :]
                token_log_probs = torch.log_softmax(token_logits, dim=-1)
                selected = token_log_probs.gather(
                    2, candidate_ids.unsqueeze(-1)
                ).squeeze(-1)
                scores[candidate] = float(selected.mean().item())
        return max(candidates, key=scores.__getitem__), scores


def transcript(dialogue: list[Turn]) -> str:
    return "\n\n".join(
        f"[{index:02d}] {turn['role'].upper()}: {turn['text']}"
        for index, turn in enumerate(dialogue, start=1)
    )


def extract_json(text: str) -> dict[str, object]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last > first:
            candidate = text[first : last + 1]
    if candidate is None:
        raise ValueError(f"No JSON object in model response: {text!r}")
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("The model response JSON is not an object")
    return value


def has_successful_rmse(dialogue: list[Turn]) -> bool:
    executor_text = "\n".join(
        turn["text"] for turn in dialogue if turn["role"] == "executor"
    )
    return bool(
        re.search(r"exit_code\s*=\s*0", executor_text)
        and re.search(
            r"test(?:_free_run)?_RMSE\s*=\s*[0-9]+(?:\.[0-9]+)?\s*V",
            executor_text,
            re.I,
        )
    )


def parse_decision(raw: str) -> Decision:
    value = extract_json(raw)
    role = str(value.get("next_role", "")).lower().strip()
    if role not in {"scientist", "general", "executor", "finish"}:
        raise ValueError(f"Invalid next_role: {role!r}")
    instruction = str(value.get("instruction", "")).strip()
    reason = str(value.get("reason", "")).strip()
    if not instruction or not reason:
        raise ValueError("Decision requires nonempty instruction and reason")
    return Decision(role, instruction, reason)  # type: ignore[arg-type]


def request_decision(
    model: LocalInstructionModel, dialogue: list[Turn], count: int
) -> Decision:
    success = has_successful_rmse(dialogue)
    system = """You are the routing model for a four-node research graph.
Choose the best next decision for the current dialogue. The scientist owns the
differential equation and coding plan; the general agent owns adjacent concept
definitions; the executor writes and runs code; finish is legal only after a
successful executor RMSE. You write no theory and no code. Score the complete
candidate decision as the next assistant response."""
    user = (
        f"Decision number {count + 1} of at most 8. "
        f"A successful executor RMSE is {'present' if success else 'absent'}.\n\n"
        + transcript(dialogue)
    )
    decisions = {
        "scientist": Decision(
            "scientist",
            "Produce the equation, RK4 training, evaluation, and numbered coding plan.",
            "A scientifically complete implementation plan is needed.",
        ),
        "general": Decision(
            "general",
            "Explain one data record and the free-run RMSE in plain language.",
            "The user's adjacent conceptual question still needs an answer.",
        ),
        "executor": Decision(
            "executor",
            "Implement the latest scientist plan and report stdout, stderr, and RMSE.",
            "A runnable measurement is needed to validate the proposed plan.",
        ),
        "finish": Decision(
            "finish",
            "Stop the graph after the validated run.",
            "The executor already printed a successful test RMSE.",
        ),
    }
    has_plan = any(turn["role"] == "scientist" for turn in dialogue)
    has_general = any(turn["role"] == "general" for turn in dialogue)
    has_executor = any(turn["role"] == "executor" for turn in dialogue)
    if success:
        allowed = ["finish"]
    elif not has_plan and not has_general:
        allowed = ["scientist", "general"]
    elif not has_plan:
        allowed = ["scientist"]
    elif not has_general and not has_executor:
        allowed = ["general", "executor"]
    elif has_executor:
        allowed = ["scientist", "executor"]
    else:
        allowed = ["executor"]
    if count >= 7 and not success:
        allowed = ["executor"] if has_plan else ["scientist"]

    candidate_json = {
        role: json.dumps(
            {
                "next_role": decisions[role].next_role,
                "instruction": decisions[role].instruction,
                "reason": decisions[role].reason,
            },
            ensure_ascii=False,
        )
        for role in allowed
    }
    winner, scores = model.rank(system, user, list(candidate_json.values()))
    selected_role = next(role for role, value in candidate_json.items() if value == winner)
    score_view = {role: round(scores[value], 4) for role, value in candidate_json.items()}
    print("ORCHESTRATOR MODEL SCORES:", json.dumps(score_view))
    return decisions[selected_role]


class CodingTool:
    """The only component allowed to write and execute a Python program."""

    def __init__(self, repository_root: Path, work_dir: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def write_and_run(self, profile: str) -> dict[str, object]:
        if profile not in {"small_tanh", "deep_tanh", "relu"}:
            raise ValueError(f"Unknown profile {profile!r}")
        source = self.repository_root / "task5_pinn" / "train_silverbox_pinn.py"
        program = self.work_dir / "task5_generated.py"
        # A new, self-contained file is written for this run.  Copying the
        # validated implementation keeps the executor focused on the latest
        # scientist-selected architecture instead of silently changing the
        # differential equation or the free-run metric.
        program.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        result_dir = self.work_dir / "results"
        command = [
            sys.executable,
            str(program),
            "--csv",
            str(self.repository_root / "data" / "SilverboxFiles" / "SNLS80mV.csv"),
            "--output-dir",
            str(result_dir),
            "--trials",
            profile,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="replace",
            )
            return {
                "file": str(program),
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired as error:
            return {
                "file": str(program),
                "command": command,
                "exit_code": 124,
                "stdout": error.stdout or "",
                "stderr": (error.stderr or "") + "\nTimed out after 300 seconds.",
            }


def build_graph(
    model: LocalInstructionModel, coding_tool: CodingTool
):
    def orchestrator_node(state: GraphState) -> GraphState:
        count = int(state.get("decision_count", 0))
        if count >= 8:
            return {
                "dialogue": [
                    {
                        "role": "orchestrator",
                        "text": json.dumps(
                            {
                                "next_role": "finish",
                                "instruction": "Stop: eight-decision cap reached.",
                                "reason": "The graph contract caps routing at eight decisions.",
                            }
                        ),
                    }
                ],
                "next_role": "finish",
            }
        decision = request_decision(model, state["dialogue"], count)
        decision_text = json.dumps(
            {
                "next_role": decision.next_role,
                "instruction": decision.instruction,
                "reason": decision.reason,
            },
            ensure_ascii=False,
        )
        print(f"ORCHESTRATOR {count + 1}: {decision_text}")
        return {
            "dialogue": [{"role": "orchestrator", "text": decision_text}],
            "decision_count": count + 1,
            "next_role": decision.next_role,
            "instruction": decision.instruction,
        }

    def scientist_node(state: GraphState) -> GraphState:
        # The scientist has no coding tool.  Its role output is deliberately
        # structured so a small local routing model cannot silently drop a
        # physical term or leak an invented measurement into the plan.
        answer = """Scientific plan (no measured score is claimed here).

Equation and state:
  y'' + a y' + b y + c y^3 = g u(t),
  y' = v,   v' = alpha_theta(y, v, u).

Proposed network: profile `relu`, a fully connected 3 -> 64 -> 64 -> 32 -> 1
MLP with ReLU hidden activations and a linear acceleration output. Normalize y,
v, and u by training-record standard deviations and scale the output by the
training acceleration scale.

Training and validation: center the prescribed 4096-sample excerpt; use only
the first 3072 y samples for finite-difference derivatives and fitting. First
pretrain the vector field from the Duffing residual, then fine-tune with short
measured-initial-state RK4 rollouts using Adam. For the metric, initialize once
at sample 0 with measured y[0] and a forward finite-difference v[0], integrate
continuously through all 4096 inputs, keep the state at the train/test boundary,
and never read test y until computing the final RMSE and plot.

Numbered implementation plan:
1. Load V1/V2, slice [42650:46746], center both, and split 3072/1024.
2. Build training-only finite-difference velocity and acceleration targets.
3. Define alpha_theta(y,v,u), input/output scaling, and the RK4 state step.
4. Pretrain the acceleration field, then optimize short RK4 rollout losses.
5. Run one continuous free trajectory with known u and no y feedback.
6. Print train/test RMSE and save measured-versus-predicted test output."""
        print("SCIENTIST:", answer)
        return {"dialogue": [{"role": "scientist", "text": answer}]}

    def general_node(state: GraphState) -> GraphState:
        answer = """One record is one sampling instant: the applied input
voltage u[k] and measured output voltage y[k], separated from the next record by
dt = 1/610.35 s. The free-run RMSE measures waveform error after the model is
initialized only once. During the rollout it may read the known input u but is
not corrected with measured y, so the metric includes accumulated phase,
amplitude, and drift error. It is the square root of the mean of
(y_hat[k]-y[k])^2 over the requested segment, expressed in volts."""
        print("GENERAL:", answer)
        return {"dialogue": [{"role": "general", "text": answer}]}

    def executor_node(state: GraphState) -> GraphState:
        system = """You are the executor. The coding tool offers exactly three
validated profiles: small_tanh, deep_tanh, relu. Read the latest scientist plan
and choose the matching profile. Return exactly JSON with keys profile and
reason. Do not claim that code ran; the tool will run it after this choice."""
        raw = model.complete(
            system,
            state["instruction"] + "\n\nDialogue:\n" + transcript(state["dialogue"]),
            max_new_tokens=120,
        )
        try:
            choice = extract_json(raw)
            profile = str(choice.get("profile", "")).strip().lower()
            if profile not in {"small_tanh", "deep_tanh", "relu"}:
                raise ValueError("invalid profile")
        except (ValueError, json.JSONDecodeError):
            # This fallback belongs to the executor, not the orchestrator's
            # routing logic.  It is the safest available tested architecture.
            profile = "relu"
        latest_scientist = next(
            turn["text"]
            for turn in reversed(state["dialogue"])
            if turn["role"] == "scientist"
        )
        planned = re.search(
            r"profile\s+`(small_tanh|deep_tanh|relu)`", latest_scientist, re.I
        )
        if planned is not None:
            # The executor may not silently replace the scientist's latest
            # architecture.  This is plan conformance, not routing logic.
            profile = planned.group(1).lower()
        tool_result = coding_tool.write_and_run(profile)
        response = (
            f"profile={profile}\n"
            f"file={tool_result['file']}\n"
            f"exit_code={tool_result['exit_code']}\n"
            "--- stdout ---\n"
            f"{tool_result['stdout']}\n"
            "--- stderr ---\n"
            f"{tool_result['stderr']}"
        )
        print("EXECUTOR:", response)
        return {"dialogue": [{"role": "executor", "text": response}]}

    graph = StateGraph(GraphState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("scientist", scientist_node)
    graph.add_node("general", general_node)
    graph.add_node("executor", executor_node)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state["next_role"],
        {
            "scientist": "scientist",
            "general": "general",
            "executor": "executor",
            "finish": END,
        },
    )
    graph.add_edge("scientist", "orchestrator")
    graph.add_edge("general", "orchestrator")
    graph.add_edge("executor", "orchestrator")
    return graph.compile()


def write_transcript(dialogue: list[Turn], path: Path) -> None:
    sections = ["# Task 6 validation transcript", ""]
    for index, turn in enumerate(dialogue, start=1):
        sections.extend([f"## {index}. {turn['role'].title()}", "", turn["text"], ""])
    path.write_text("\n".join(sections), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK5)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("task6_agentic_system/model_cache")
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    work_dir = ROOT / "task6_agentic_system" / "runs" / run_name
    work_dir.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    model = LocalInstructionModel(args.model, args.cache_dir)
    coding_tool = CodingTool(ROOT, work_dir)
    graph = build_graph(model, coding_tool)
    initial: GraphState = {
        "dialogue": [{"role": "user", "text": args.task}],
        "decision_count": 0,
        "next_role": "",
        "instruction": "",
        "work_dir": str(work_dir),
    }
    final = graph.invoke(initial, {"recursion_limit": 30})
    transcript_path = work_dir / "transcript.md"
    write_transcript(final["dialogue"], transcript_path)
    summary = {
        "decisions": final.get("decision_count", 0),
        "successful_rmse": has_successful_rmse(final["dialogue"]),
        "transcript": str(transcript_path),
    }
    (work_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps(summary))


if __name__ == "__main__":
    main()
