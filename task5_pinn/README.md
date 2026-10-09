# Task 5 — Silverbox neural ODE

`train_silverbox_pinn.py` trains a fully connected acceleration model
`alpha_theta(y, v, u)` and integrates `(y, v)` with RK4.  The required
4096-sample excerpt is centered, split 3072/1024, and evaluated by one
continuous free run.

From the repository root:

```powershell
py task5_pinn/train_silverbox_pinn.py
```

The script compares three architectures, writes checkpoints and predictions to
`task5_pinn/results`, prints both free-run RMSE values, and creates the test
plot used in the solution report.
