# PhotonicsAI Homework 1

Complete solutions and reproducible code for Tasks 1–6.

## Deliverables

- `deliverables/homework-1-solutions.pdf` — the submission report.
- `deliverables/homework-1-solutions.docx` — editable source document.
- `task5_pinn/train_silverbox_pinn.py` — Silverbox neural ODE/PINN experiment.
- `task6_agentic_system/agentic_task5.py` — four-node LangGraph system.

## Data

Download the official Silverbox archive from
<https://www.nonlinearbenchmark.org/benchmarks/silverbox> and place
`SNLS80mV.csv` at `data/SilverboxFiles/SNLS80mV.csv`.

## Task 5

```powershell
py task5_pinn/train_silverbox_pinn.py
```

The final comparison uses seed 17. The kept `3→64→64→32→1` ReLU model prints:

- train free-run RMSE: `0.00106542 V`
- test free-run RMSE: `0.00311170 V`
- zero-output test baseline: `0.06049180 V`

## Task 6

Create the environment and install the listed requirements, then run:

```powershell
.\.venv\Scripts\python.exe task6_agentic_system\agentic_task5.py
```

The validated run is recorded in
`task6_agentic_system/runs/validation_run_5/transcript.md`. The graph made four
orchestrator decisions and the executor completed with exit code 0 and the same
test RMSE, `0.00311170 V`.
