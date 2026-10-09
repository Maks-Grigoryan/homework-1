# Task 6 — LangGraph agentic system

The program builds a `StateGraph` with four nodes:

- `orchestrator`: a model-generated structured routing decision;
- `scientist`: the equation, experiment design, and coding plan;
- `general`: adjacent conceptual explanations;
- `executor`: the only role that receives the write-and-run coding tool.

Every node appends a named turn to the shared dialogue. Specialist edges return
to the orchestrator. The orchestrator stops only after a successful executor
RMSE or after eight decisions.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe task6_agentic_system\agentic_task5.py
```

The default LLM is `Qwen/Qwen2.5-0.5B-Instruct`, run locally. Its first run
downloads model weights into `task6_agentic_system/model_cache`. Each validation
run gets a new directory containing the generated Python file, captured output,
metrics, plot, and complete dialogue transcript.
