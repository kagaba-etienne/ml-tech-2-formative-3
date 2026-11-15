Did experiment 1 (Base)

# hyperparameters
LR=1e-4,
GAMMA=0.99,
BATCH_SIZE=32,
EPS_START=1.0,
EPS_END=0.05,
EPS_DECAY=50000,
# training loop hyperparameters
MEMORY_CAPACITY=100000,
TARGET_UPDATE_FREQ=1000,
NUM_EPISODES=500

The model did not learn, the ma was flat for both rewards and steps/episode


2 experiments underway

1 on kaggle t4 and another on colab t4

each same run except on colab I updated to only do gradient descent after every 4 steps instead of every step
if agent.steps_done % 4 == 0:
    agent.optimize_model(memory)
    

Other possible optimizations I am thinking of:

- increasing the batchsize from 32 to 64
- moving the scaling from cpu to gpu (reduces data footprint on the cpu and transfer is faster)


The colab failed to run because the code exhausted the RAM available. I moved the scaling to happen on gpu to reduce the memory footprint on the ram. The run is still running.work fits the replay data less aggressively, which reduces the risk of overfitting to recent experience. This is the part that resembles regularization.

With mse loss, I saw that the base model did not show any signs of learning within the number of episodes we were running. I switched to huberloss.

The colab finished relatively faster than the one on kaggle cause of the optimizations. and one of the optmizations (the gradient calculation after every 4 steps) turned to be a stabilizer of the network and we can see that the learning is starting to have an upward trend.


Now I can see that the waiting improved stability, but it does not help me rule out that the stabilization was also facilitated by huber loss.

I am running two more trainings one with mse with the optmizations. and another one with mse + optmizations + reward clipping.

The one with just optmizations finished running, and the results show no signs of learning as before. It seems like the model is becoming conservative (learned a policy that is not good), as the duration in the game is still overall the same but the rewards it gains in the game are reducing.

It seems like the large updates in just mse were the reason behind its inability to learn. Strong/large TD error would dominate the whole batch, and the model could not learn anything useful. but now with clipped rewards, it is able to achieve much higher rewards even than the huber loss variant.

""" need to look at this and understand how huge TD errors are a problem.

With unclipped rewards and MSE, a few huge TD errors dominate each batch, and the Q-values tend to inflate through the feedback loop we talked about: overestimated values feed into the targets, which produces more overestimation. When that happens, one or two actions end up with inflated Q-values and the greedy policy just repeats them. Something like staying in a safe spot without lining up shots would fit your curves: survival stays about the same and scoring collapses.
"""

I also run a huber + fast + clipping, but it is performing slightly similar to the mse vairant even though the mse has a slightly better performance than huber.This difference might be be due to the seed variance since each training was run once. For further hyper parameter tuning, we will go with mse variant.


Each of the fast runs took about 19-20 mins to run


Two runs are underway:
1. mse variant with larger learning rate of 1e-3
2. mse variant with a smaller gamma of 0.8

The two runs finished but none of them is stronger than the mse variant with default parameters nor the huber variant. (each of these runs took 10 mins)

Going to try the same runs but using huber variant to see if the loss function has any effect. 

The runs have completed (8min & 11mins of execution). Also trying the same hyperparameters on the huber variant reduced the model performance proving that reducing learning rate or gamma does not yeild any improvement on the model rather they hurt model performance.

The next two runs I will try increasing the batch size on both huber and mse from 32 to 64.
The two runs are underway!

The mse variant performed significantly better and is the best model so far. The huber variant did not see significant improvement.

Now going to try increasing the batchsize from 64 to 128 on the mse variant and see what happens.

One run is under way!

It finished, the run is comparable to the one with 64 batch size and it has the highest reward among all models (6000+) even though I still consider the one with 64 batch size a better model overall.
