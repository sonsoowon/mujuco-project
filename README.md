# RL Project
Final Project of Reinforcement Learning

### 1. Environment : Mujuco
||Ant-v4|HalfCheetah-v4|
|:---:|:---:|:---:|
|Action Space|Box(-1.0, 1.0, (8,), float32)|Box(-1.0, 1.0, (6,0), float32)|
|Observation Space|Box(-inf, inf, **(27,)**, float64)|Box(-inf, inf, **(17,)**, float64)|

<br>

### 2. Network Structure & RL Techniques
![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/a673ce4c-5e54-4f6b-8d3e-76aca63a41b1)

1. WIDER value network than policy network
2. depth=1 or 2
3. activation function: Tanh (ReLU is WORST)
4. Orthogonal Initialization
5. Normalization for every layer of value network
6. Advantage Normalization

<br>


### 3. Results


- **Ant-v4**
  
  ||Non-Normalize Observation|Normalize Observation|
  |---|:---:|:---:|
  |Train|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/8d7810da-6b59-4d03-9812-a554826c8594)|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/16c1f3b4-3b9a-41ce-96d5-a6f899ea0c14)|
  |Test|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/a2ba51b9-2e3b-40e4-97e9-b5f1aaad8f98)|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/7cbffcaf-1b5b-40d9-aa55-8075b3fdbb0f)|

<br>

- **HalfCheetah-v4**
  
  ||Non-Normalize Observation|Normalize Observation|
  |---|:---:|:---:|
  |Train|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/2c2bce30-3e4c-41ab-9d21-7cb471097801)|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/b678a303-23df-4d69-ab71-0c96a516192d)|
  |Test|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/ca497b98-fe7c-4d58-b575-37136978e34d)|![image](https://github.com/sonsoowon/mujuco-project/assets/55790232/1ecb2457-f4a4-45a3-aa0d-b7cc8275c027)|

<br>

- **Test Video**

  |Ant-v4|HalfCheetah-v4|
  |:---:|:---:|
  |<video src="https://github.com/sonsoowon/mujuco-project/assets/55790232/4f46eb6f-1853-439e-8d42-8df18f552ee5">|<video src="https://github.com/sonsoowon/mujuco-project/assets/55790232/d3037681-f47f-4b5b-b594-b873bd20da55">|

<br>

### 4. Conclusion
- Return oscillates more loudly in Ant-v4
  <br>&rarr; Bigger Observation Space, Greater Oscillation Width

- Observation nomalization improved performance of Ant-v4, but reduced performence of HalfCheetah-v4
  <br>&rarr; Observation normalization can reduce performance when the observation space is small

<br>

#### Reference
- [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf)
