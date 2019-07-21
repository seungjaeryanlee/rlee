# rlee

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/rlee.svg?label=black)](https://travis-ci.com/seungjaeryanlee/rlee)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/rlee.svg?label=flake8)](https://travis-ci.com/seungjaeryanlee/rlee)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/rlee.svg?label=isort)](https://travis-ci.com/seungjaeryanlee/rlee)
[![mypy Build Status](https://img.shields.io/travis/com/seungjaeryanlee/rlee.svg?label=mypy)](https://travis-ci.com/seungjaeryanlee/rlee)
[![pytest Build Status](https://img.shields.io/travis/com/seungjaeryanlee/rlee.svg?label=pytest)](https://travis-ci.com/seungjaeryanlee/rlee)

[![numpydoc Docstring Style](https://img.shields.io/badge/docstring-numpydoc-blue.svg)](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue.svg)](.pre-commit-config.yaml)

Rlee (pronounced "early") is a research framework built on top of PyTorch 1.0 for fast prototyping of novel reinforcement learning algorithms. These small, easily grokked implementations will make it easier for researchers to build research on top of them.

## Papers

### Implemented

- [x] [[DQN] Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://deepmind.com/research/dqn/)
- [x] [[Double DQN] Deep Reinforcement Learning with Double Q-learning (Hasselt et al., 2015)](https://arxiv.org/abs/1509.06461)
- [x] [[Combined ER] A Deeper Look at Experience Replay (Zhang and Sutton, 2017)](https://arxiv.org/abs/1712.01275)

### Will be Implemented

- [ ] [[Prioritized ER] Prioritized Experience Replay (Schaul et al., 2015)](https://arxiv.org/abs/1511.05952)
- [ ] [[Dueling DQN] Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2015)](https://arxiv.org/abs/1511.06581)
- [ ] [[Distributional DQN] A Distributional Perspective on Reinforcement Learning (Bellmare, Dabney, and Munos, 2017)](https://arxiv.org/abs/1707.06887)
- [ ] [[Rainbow] Rainbow: Combining Improvements in Deep Reinforcement Learning (Hessel et al., 2017)](https://arxiv.org/abs/1710.02298)
- [ ] [[DQfD] Deep Q-learning from Demonstrations (Hester et al., 2017)](https://arxiv.org/abs/1704.03732)

### Future Candidates

- [ ] [[Boostrapped DQN] Deep Exploration via Bootstrapped DQN (Osband et al., 2016)](https://arxiv.org/abs/1602.04621)
- [ ] [[DRQN] Deep Recurrent Q-Learning for Partially Observable MDPs (Hausknecht and Stone, 2015)](https://arxiv.org/abs/1507.06527)
- [ ] [[Bayesian DQN] Efficient Exploration through Bayesian Deep Q-Networks (Azizzadenesheli and Anandkumar, 2018)](https://arxiv.org/abs/1802.04412)
- [ ] [[QR-DQN] Distributional Reinforcement Learning with Quantile Regression (Dabney et al., 2017)](https://arxiv.org/abs/1710.10044)
- [ ] [[IQN] Implicit Quantile Networks for Distributional Reinforcement Learning (Dabney et al., 2018)](https://arxiv.org/abs/1806.06923)
- [ ] [[AIQN] Autoregressive Quantile Networks for Generative Modeling (Ostrovski, Dabney, and Munos, 2018)](https://arxiv.org/abs/1806.05575)
- [ ] [[LS-DQN] Shallow Updates for Deep Reinforcement Learning (Levine et al., 2017)](https://arxiv.org/abs/1705.07461)
- [ ] [[DQN-ITS] Information-Directed Exploration for Deep Reinforcement Learning (Nikolov et al., 2018)](https://arxiv.org/abs/1812.07544)
- [ ] [[DQN+CTS] Unifying Count-Based Exploration and Intrinsic Motivation (Bellmare et al., 2016)](https://arxiv.org/abs/1606.01868)
- [ ] [[DQN+PixelCNN] Count-Based Exploration with Neural Density Models (Ostrovski et al., 2017)](https://arxiv.org/abs/1703.01310)
