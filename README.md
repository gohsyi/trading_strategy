# Trading Strategy

Course project of SJTU EE359 Data Mining (advised by Prof. Bo Yuan), where we use reinforcement learning to decide trading strategy. This is the repo for Task 3. For Task 1 and 2, please take a look at my partner Ruofan Liang's repo [here](https://github.com/nexuslrf/Financial-DataMining).

## Task: Trading strategy based on reinforcement learning

This task involves reinforcement learning and will no longer use the tags in task 1. To simplify the problem, this task sets that each tick has at most 5 hand long positions and 5 hand short positions. Long positions and short positions cannot be held at the same time. A tick can only have one action at a time. Positions can be increased or decreased (with unit equals one hand) through buying and selling, and the absolute value of change in the number of positions of one action cannot exceed one hand. The current state can be maintained by an idle action. When the buying action is executed, the purchase will be successful and will not have any impact on the market. The price is AskPrice1 of the current tick. When the selling action is executed, the sell will be successful and will have no effect on the market. The price is BidPrice1 of the current tick. Finally, you should include in your report:
the number of buying and spelling on testing set
the average price to buy and the average price to sell.
Besides, attach action selection for each tick on testing set for submission.

Here are a few typical models and algorithms for reference:
Deep Direct Reinforcement Learning for Financial Signal Representation and Trading.
Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks.


## Run

The entry is `main.py`. You can run following command in you terminal.

```
python main.py \
-train_path path/to/the/training/data
-test_path path/to/the/validation/data
-load_path path/to/the/ckpt/file/you/want/to/finetune/or/test
-pred_path path/to/the/prediction/results
-total_epoches {total number of episodes, 0 if you want to test the performance of an existing ckpt}
```

for more arguments, please refer to `common/argparser.py`.


## Reinforcement Learning Model

We adopt A2C RL model to make trading decisions for us. The framework of this model is introduced above. Next, we introduce how to implement it on our task by giving the definition of _observation_, _action_, _reward_ and how to _update_ it using gradients.

### Observation
The agent makes its actions only based on the observation it gets. The observation in our setting is price information in the past $10$ steps and future price prediction.

### Action
The action space is {-1, 0, 1}, denoting short hand (sell), idle, long hand (buy) correspondingly. So the actor network has three outputs, representing the probability of taking each of these three actions. Then we can sample an action with these probabilities.

Notice that the observation is a time series, so the actor network has an RNN architecture. We input the final output to an MLP and get the policy.

Also, when the agent makes some unavailable action, such as buying when holding 5 shares or selling when holding 0. The environment will treat the action as idle, so the state remains. 

### Reward

The reward `r(a_t, s_t)` is defined as `a_t * (p_{t+1} - p_t)`. Intuitively, the agent gets a positive reward when it sells before the price goes down or buys before the price goes up and the magnitude is proportional to the amount of price change.

The update follows the A2C policy iteration process. We use gradient descent to find the optimal policy parameter. So the process is
```
\theta = \theta + \eta \frac{\partial}{\partial \theta} \log \pi(a | s;\theta) A(s, a)
```
where `\eta` is the learning rate.


## Baseline Models

### Stochastic Model
The simplest stochastic model is stochastic model, which just randomly select an action among _short position_, _idle_, and _long position_. 

### Rule-based Supervised Learning Model
This model is based on the rule that selling when predicting the price going down and buying when predicting the price going up. This way, we can either develop new model to predict the future price or just adopt the models we trained on task 1.

Also, in baseline models, when the agent makes some unavailable action, such as buying when holding 5 shares or selling when holding 0. The environment will treat the action as idle, so the state remains. 

This model is a supervised learning model because the predicting model usually uses supervised learing, which we can adopt _XGBoost_, _DNN_, _LSTM_, _CNN_ and _Transformer_. These models are introduced in my partner Ruofan Liang's repo [here](https://github.com/nexuslrf/Financial-DataMining).


## Experiments

First, we give the results that the relative number (proportion) of buying and selling on test set in the table below. Both most and least active buying and selling are made by rule-based supervised learning model.

<img src="https://github.com/gohsyi/trading_strategy/blob/master/figures/proportion.png" height="225" width="600"/>

Then, we report the average price to buy and the average price to sell in the table below, where it's not hard to notice that the rule-based supervised learning model achieves better performance than stochastic model and A2C model no matter what model is used to predict the future price.

<img src="https://github.com/gohsyi/trading_strategy/blob/master/figures/average_price.png" height="300" width="600"/>

We illustrate the total assets during the whole trading process in figures below.

<img src="https://github.com/gohsyi/trading_strategy/blob/master/figures/xgboost.png" height="300" width="750"/>

<img src="https://github.com/gohsyi/trading_strategy/blob/master/figures/dnn.png" height="600" width="750"/>


### Conclusion

Based on the above experiment results, we can compare our trading models as following:
The stochastic model performs the poorest. The rule-based supervised learning model performs the best. Our Reinforcement Model is between them.

So, why bother using Reinforcement Learning model since it cannot outperform the rule-based supervised learning model? We give the following reasons.

1. First, the trading rule is absolutely not the best strategy we can have. A simplest deficiency is that, we do not have to buy the share when the predicted price goes up. First, the prediction is not necessarily correct. And besides, considering the total long/short positions are limited to $5$, it may be better to choose the buying and selling time more carefully. However, reinforcement learning can do this job well as long as we feed sufficient data into it and use network with proper architecture and proper number of parameters.
    
2. In addition, the rule is relatively too simple, and cannot handle the more complex situation in the market. For example, in real world, we can buy or sell more than 1 unit of shares at a time and the bidding price and asking price are more complex, too. For rule-based model, we need to re-design the rule and it may become too complex and troubling. But for reinforcement learning model, we can just modify the action space and tune parameters to solve the problem. So reinforcement learning model has stronger generalization ability on this task especially in real world.
