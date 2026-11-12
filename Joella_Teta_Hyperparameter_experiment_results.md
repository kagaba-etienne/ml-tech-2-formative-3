# Hyperparameter Experiment Results – Joella Teta

**Environment:** Atari Air Raid (`ALE/AirRaid-v5`), DQN with PyTorch and Gymnasium
**Setup:** Every experiment trained for 200 episodes with Adam, replay buffer 100,000, target network update every 1,000 steps. Only the values in the table were changed. Each experiment was run once.
**Metric:** "Avg reward" is the mean episode reward over the last 50 episodes. "Best" is the highest single-episode reward.

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 50000 | Baseline. Peaked in episodes 51–100 (avg 744) then slipped to 566. Did not keep improving. |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 50000 | Slow, steady rise (576 to 673) with very large swings between episodes. Higher learning rate did not make training collapse. |
| 3 | 2.5e-5 | 0.99 | 32 | 1.0 | 0.05 | 50000 | Strong middle phase (832 in episodes 51–100), ended at 726. Slow learning rate still performed well. |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 50000 | Gradual, steady improvement (611 to 690). Slightly better than baseline. |
| 5 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 50000 | Worst run. Reward fell every block (592 to 338) and episodes got shorter. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 50000 | Rose to 713 then dipped to 650. Similar to baseline. |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 50000 | Best run. Improved in every block (530 to 852); last 20 episodes averaged 915. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 10000 | Fast epsilon decay. Improved from 560 to 698, slightly above baseline. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Slow epsilon decay. Flat (about 570 throughout, 532 at the end); too much random exploration for 200 episodes. |
| 10 | 5e-4 | 0.95 | 64 | 1.0 | 0.01 | 20000 | Combination run. Rose to 740 then settled at 714. Highest single episode of all runs (3375). |

## Detailed observations

Average reward by block of 50 episodes (episodes 1–50, 51–100, 101–150, 151–200):

| Experiment | Block 1 | Block 2 | Block 3 | Block 4 (last 50) | Last 20 | Best episode |
|---|---|---|---|---|---|---|
| 1 | 672 | 744 | 627 | 566 | 500 | 2225 |
| 2 | 576 | 621 | 696 | 673 | 600 | 2575 |
| 3 | 617 | 832 | 650 | 726 | 768 | 2225 |
| 4 | 611 | 618 | 666 | 690 | 708 | 2325 |
| 5 | 592 | 431 | 350 | 338 | 308 | 1700 |
| 6 | 630 | 684 | 713 | 650 | 656 | 2550 |
| 7 | 530 | 595 | 748 | 852 | 915 | 2775 |
| 8 | 560 | 694 | 650 | 698 | 632 | 2025 |
| 9 | 570 | 570 | 586 | 532 | 406 | 2725 |
| 10 | 599 | 622 | 740 | 714 | 622 | 3375 |

### Experiment 1 – Baseline (LR 1e-4, gamma 0.99, batch 32, decay 50000)
The agent learned a little in the first 100 episodes, then its average reward drifted down. The final 50-episode average (566) is below the average of episodes 51–100 (744), so training was not stable enough to keep improving in 200 episodes.

### Experiment 2 – Higher learning rate (1e-3)
Reward climbed slowly from 576 to 673 and the episode scores swung widely (standard deviation about 490 in the last 50 episodes). The larger updates made play noisier, but training did not collapse.

### Experiment 3 – Lower learning rate (2.5e-5)
This run had the best middle phase of the learning-rate group (832) and finished at 726. A smaller step size gave smoother learning here. With one run per setting, this may be partly luck.

### Experiment 4 – Gamma 0.95
Steady upward trend and 690 at the end, above the baseline. Slightly shorter planning horizon did not hurt.

### Experiment 5 – Gamma 0.90
The clearest failure. Reward fell in every block and the average episode length dropped to about 308 steps (about 620 for the baseline). A low gamma makes the agent care less about future rewards, which hurts in a game where survival and shooting over time matter.

### Experiment 6 – Batch size 64
Improved until episodes 101–150 (713) and then dipped. Overall close to the baseline.

### Experiment 7 – Batch size 128
The only run that improved in every block, ending at 852. Larger batches average more samples per update, so the gradient is less noisy and learning is steadier. This was the best configuration of all 10.

### Experiment 8 – Fast epsilon decay (10,000)
The agent stopped exploring early and began using its learned policy sooner. It ended at 698, above the baseline, so less exploration did not hurt over 200 episodes.

### Experiment 9 – Slow epsilon decay (100,000)
Reward stayed flat (about 570) and fell to 532 at the end. The agent was still taking many random actions when training stopped, so it had little time to benefit from what it learned.
### Experiment 10 – Combination (LR 5e-4, gamma 0.95, batch 64, epsilon end 0.01, decay 20000)
Reward rose to 740 in episodes 101–150, then levelled off at 714. It produced the highest single episode (3375). Combining settings that looked good individually did not beat batch size 128 alone.

## Conclusions

1. **Gamma had the biggest effect.** 0.90 clearly hurt; 0.95 and 0.99 were similar.
2. **Batch size 128 gave the best and most consistent improvement.**
3. **Epsilon decay:** a very slow decay (100,000) was too slow for 200 episodes; fast decay (10,000) was fine.
4. **Learning rate showed no clear pattern** between 2.5e-5 and 1e-3. Differences were within the noise.
5. **Limits:** each setting was run once for 200 episodes, and single-episode rewards vary by hundreds of points. Only experiment 5 (clearly worse) and experiment 7 (steadily better) stand out strongly. Longer training and several seeds would be needed to be confident about the rest.

## Final hyperparameter choice
Experiment 7 (LR 1e-4, gamma 0.99, batch size 128, epsilon 1.0 to 0.05 with decay 50000) is the configuration to justify as the best.

## Files
- Training plots: `training_plots.png` and `training_plots_per_experiment.png`
- Models and logs: `dqn_exp1_model.pth` to `dqn_exp10_model.pth`, `training_performance_exp1.csv` to `training_performance_exp10.csv`
