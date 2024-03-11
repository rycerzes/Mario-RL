# MLSA-Mario
![MLSA Logo](https://github.com/preasha07/MLSA_webdev/assets/150994559/1f3dcf4c-ceb6-442e-8b2c-eb7b68fbb129)

<a>
  <h1 align="center">Mario-RL</h1>
</a>

<p align="center">
  Project Synopsis<br>
  Project stack in short
</p>

## 🚧Our Project :

Our project encompassed training an AI agent to master Super Mario Bros using reinforcement learning , Python, and the nes-py emulator. After establishing the development environment and integrating the AI agent with the emulator, we have employed reinforcement learning algoirthms. This enabled the AI to learn strategies via trial and error, dynamically adjusting to game obstacles.

The AI agent, designed for sequential decision-making, dynamically tailors its strategies to overcome in-game obstacles and opponents. This projects serves as a compelling testament to the promise of reinforcement learning in developing intelligent agents for autonomous gameplay, particularly in classic video games like Super Mario Bros. We have given our special attention to Super Mario Bros and it's emulation via nes-py while the building and implementation of our project.

## 🖥️Project Stack :

   `Python 3.11` : Widely used for machine learning and reinforcement learning projects.  
- **Reinforcement Learning Libraries:**

   `Gymnasium` : Provides the environment and a toolkit for developing and comparing reinforcement learning algorithms.

   `Stable-Baselines3` : A set of high-quality implementations of reinforcement learning   algorithms in Python, built on top of Gymnasium.
  
- **Emulation and Interaction:**
    
    `nes-py` : A Python-based Game Boy emulator that allows interaction with the Super Mario Bros ROM. It provides a platform for the AI agent to play the game.
    
- **Machine Learning Framework:**
    
    `TensorFlow`  or `PyTorch` : Choose one of these deep learning frameworks to implement  neural networks for reinforcement learning algorithms.
    
- **ROM Access:**
    
    `Super Mario Bros ROM` : The original game ROM file, which serves as the input for the nes-py emulator. Ensure legal and ethical use of the ROM.
    
- **Development Environment:**
    
    Jupyter Notebooks or IDE (e.g., VSCode): For writing, testing, and debugging code.
    
- **Version Control:**
    
    `Git` : For version control and collaboration.
    
- **Data Visualization:**
    
    `Matplotlib`  or `Seaborn` : For visualizing training progress, performance metrics, and other relevant data.


## Features

- **Reinforcement Learning Mastery**- Mastered the art of applying reinforcement by learning principles to the classic game Super Mario Bros.
- **Problem-Solving Prowess**- Improved problem-solving skills by overcoming game challenges such as obstacle navigation and enemy interaction.
- **AI Agent Autonomy**- Our autonomous AI agent skillfully navigates and completes levels in a complex gaming environment, showcasing the power of reinforcement learning.
- **Algorithm Understanding**- We've deepened our knowledge of RL algorithms like PPO by applying them to train smart agents.
- **Adaptability & Sequential Decision-Making**- Our AI agent shows off adaptability and sequential decision-making skills, both critical in dynamic game scenarios.
- **Nes-py Integration Expertise**- We have become proficent in intergrating and using the nes-py emulator, a handy skill for Game Boy projects.
- **Visualization & Analysis Capabilities**- We have grown our abilities to visualize and analyse training progress, a fundamental part of monitoring and enhancing AI models.
- **Educational Showcase Creation**- We have put together an educational showcase that makes RL concepts and their application visually engaging.
- **Collaboration & Documentation Improvement**- Our collaboration and documentation skills have improved by sharing code, insights, and results on platforms like GitHub.
- **Model Deployment**- Thrrough deploying the model, we have learned valuable lessons about model deployment for real-time or interactive gameplay scenarios.

## Overview

The Super Mario Bros Reinforcement Learning project presents an in-depth, multi-faceted learning opportunity. It skillfully blends theoretical comprehension with practical application. This project goes beyond just grasping concepts- it's about employing those concepts in real-life situations. It serves as an ideal stage for refining skills with broad applications in the fast-growing realms of AI and ML. By engaging in this project, participants can immerse themselves in these intricate subjects and gain a distinctive combination of knowledge and hands-on skills invaluable in today's world.

## Running locally

First, run the development server:

```bash
poetry update
python main.py
```

This should start a terminal session which will show all values for during training. It will also spawn a game window in which one can see the agent playing Mario throughout the different sessions

> [!IMPORTANT]  
> Important note 

> [!NOTE]
> Document note

## Contributors

| CONTRIBUTORS | MENTORS | CONTENT WRITER |
| :------:| :-----:| :-----: |
| Ahona Bar| Swapnil Dutta | Preasha Ghoshal |
| Aman Kumar Anubhav | | |
| Prasoon Modi | | |
| Romit Chatterjee| | |
| Sainath Dey | | |
| Shourya Merchant | | |
| Soham Roy | | |
| Twinkle Pal | | |



## Version
| Version | Date          		| Comments        |
| ------- | ------------------- | --------------- |
| 1.0     | 08-03-2024 | Initial release |

## Future Roadmap
- [ ] Better evaluation after agent training
- [ ] Using Config file for easy changing of hyperparameters
- [ ] Visualization and Metrics
- [ ] Better Error Handling

## Acknowledgements
DDQN - https://arxiv.org/abs/1509.06461

PyTorch RL Super Mario Bros Example - https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
