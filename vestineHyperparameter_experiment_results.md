# vestine — Hyperparameter experiment results

Environment: `ALE/AirRaid-v5` | Episodes per experiment: 50

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50000 | Baseline: mean reward 742. Reward fell from 577 (1st half) to 576 (2nd half). Best episode 1525. |
