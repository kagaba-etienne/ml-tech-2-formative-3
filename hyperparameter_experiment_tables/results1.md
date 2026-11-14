Member Name: **Vestine Umukundwa**

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Baseline. Learning was variable but improved overall; final reward was 375, with a last-20-episode mean of 546.2 and a maximum of 1,675. |
| 2 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Higher learning rate produced stronger rewards than the baseline; final reward was 600, last-20 mean was 711.2, and maximum reward was 2,150. |
| 3 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Lower learning rate still learned, but improvement was slower and less consistent; final reward was 725, last-20 mean was 557.5, and maximum reward was 1,750. |
| 4 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 50,000 | Lower gamma gave a high late-run average (707.5) and the highest peak reward (2,450), but the final reward fell to 275, showing instability. |
| 5 | 0.0001 | 0.995 | 32 | 1.0 | 0.05 | 50,000 | Higher gamma resulted in weaker and less stable performance; final reward was 275, last-20 mean was 416.2, and maximum reward was 1,500. |
| 6 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 50,000 | Smaller batches produced the strongest average performance; last-20 mean was 867.5 and maximum reward was 2,550, although rewards fluctuated. |
| 7 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 50,000 | Larger batches learned reasonably well and ended at 850, but the last-20 mean was moderate at 552.5 and the maximum was 1,675. |
| 8 | 0.0001 | 0.99 | 32 | 0.8 | 0.05 | 50,000 | Reduced initial exploration gave a strong late average (691.2) but inconsistent rewards; final reward was 250 and maximum reward was 1,925. |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 50,000 | Lower final exploration produced good late performance; final reward was 775, last-20 mean was 671.2, and maximum reward was 3,050. |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Slower epsilon decay maintained exploration longer and produced the highest final reward, 1,625; last-20 mean was 640.0 and maximum reward was 1,925. |

