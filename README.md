[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

- 김영훈 (한양대학교ERICA 인공지능학과)
 
## Part 1. Introduction to Reinforcement Learning

- [P01-1. Introduction to Reinforcement Learning (pytorch).pdf](https://github.com/user-attachments/files/24773269/P01-1.Introduction.to.Reinforcement.Learning.pytorch.pdf)
- [P01-2_backpropagation - upload.pdf](https://github.com/user-attachments/files/16246790/P01-2_backpropagation.-.upload.pdf)
- [P1. atari-breakout-cuda.zip](https://github.com/user-attachments/files/21137927/P1.atari-breakout-cuda.zip)

<!--
- [P1. atari-breakout-cuda.ipynb - Colab.pdf](https://github.com/user-attachments/files/18506054/P1.atari-breakout-cuda.ipynb.-.Colab.pdf)
- [pytorch-dqn-atari-practice.zip](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269606/pytorch-dqn-atari-practice.zip)
-->
<!--
```python
import torch
from torch.autograd import Variable

x = Variable(
 torch.tensor(1., dtype=torch.float32),
 requires_grad=True)
y = Variable(
 torch.tensor(1., dtype=torch.float32),
 requires_grad=True)
z = Variable(
 torch.tensor(1., dtype=torch.float32),
 requires_grad=True)

optimizer = torch.optim.SGD(params=[x, y, z], lr=0.01)

EPOCHS = 1000
for epoch in range(EPOCHS):
    f = (x + y + z)**2 + (x-1)**2 + (y-1)**2 + (z-1)**2
    optimizer.zero_grad()
    f.backward()
    optimizer.step()
```
-->

## Part 2. Markov Decision Process (Dynamic Programming Approaches)

- [P02. Markov Decision Process - upload.pdf](https://github.com/user-attachments/files/16267407/P02.Markov.Decision.Process.-.upload.pdf)
- [example-policy-eval-gridworld.zip](https://github.com/user-attachments/files/21154317/example-policy-eval-gridworld.zip)

## Part 3. Monte-Carlo RL (Bootstrapping Approaches)

- [P03. Monte-Carlo RL - upload.pdf](https://github.com/user-attachments/files/16267413/P03.Monte-Carlo.RL.-.upload.pdf)
- [example-mc-black-env.ipynb.pdf](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269241/example-mc-black-env.ipynb.pdf) <!--[example-mc-black-env.zip](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269611/example-mc-black-env.zip)-->
- [example-mc-on-policy.ipynb.pdf](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269242/example-mc-on-policy.ipynb.pdf)
- [example-mc-off-policy.ipynb.pdf](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269244/example-mc-off-policy.ipynb.pdf)


## Part 4. Temporal-Difference RL (Bootstrapping Approaches)

- [P04. Temporal-Difference RL.pdf](https://github.com/user-attachments/files/18544330/P04.Temporal-Difference.RL.pdf)
- [P4_taxi.ipynb.zip](https://github.com/user-attachments/files/21174083/P4_taxi.ipynb.zip)

<!--
- [example-td-sarsa.ipynb.pdf](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269246/example-td-sarsa.ipynb.pdf)
- [example-td-qlearning.ipynb.pdf](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269247/example-td-qlearning.ipynb.pdf)
- [example_td_qlearning_taxi.ipynb.pdf](https://github.com/nongaussian/class-2023-skt-fly-ai/files/12269248/example_td_qlearning_taxi.ipynb.pdf)
-->

## Part 5. Deep Q-Learning

- [P05. Deep RL (torch) - upload.pdf](https://github.com/user-attachments/files/16269930/P05.Deep.RL.torch.-.upload.pdf)
- [dqn-shootingairplane-torch-dist.zip](https://github.com/nongaussian/class-2024-skt-fly-ai/files/13830379/dqn-shootingairplane-torch-dist.zip)
- [P5. box2d-lunarlander-gym.ipynb.zip](https://github.com/user-attachments/files/21172651/P5.box2d-lunarlander-gym.ipynb.zip)
- [gym_examples.zip](https://github.com/user-attachments/files/21172582/gym_examples.zip)


