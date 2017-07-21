# University of Alberta Robot Learning Project

## Background
Learning accurate knowledge through a stream of real world data, like the low level sensorimotor data found in robotic systems, present a unique challenge for any intelligent system. The data is immense and the environment continually changes. Furthermore, such a real time learning system must handle asynchronous timing issues. A robot for example, exists in a domain where it 
- receives a continuous stream of input.
- Requires computational time to make a decision about a next action
- Experiences delays between making the decision and the action actually being taken
- Experiences delays between the action taking place, and the effect actually being observed. 
This asynchronous nature presents challenges not seen in more simple "grid world" settings where actions and observations are neatly separated in time. 

Given that any agent learning and interacting with the real world will need to overcome these challenges, there is strong motivation for exploring and experimenting in this setting. However, these environments aren't easily accessible. They often require a physical robot set up to easily control and monitor - something not nearly as accessible as other learning environments.

## Project Goal
Given the desire to experiment in a setting grounded in low level sensorimotor data, and the lack of access to such settings, the goal of this project is therefore 2 fold:
1. *Environment creation:*
To create an experimental environment that makes it easy for researchers to conduct reinforcement learning (or other) experiments, grounded in real time sensorimotor data. This environment will include instructions and documention on configuring and setting up the Turtlebot 2 such that it can be controlled and monitored easily. Any researcher with access to a Turtlebot should be able to begin conducting experiments within hours. 
2. *Experiments:*
To perform some learning task using the experimental environment. In addition to contributing to knowledge in reinforcement learning and robotics, performing such a task will help achieve the prior goal of creating an experimental setup for conducting experiments. 

## Setup
In order to run the python code, the C++ code that runs the tile coding must be compiled. To compile, cd to [repo]/ROSBase/src/horde/scripts/CTiles/. Then, execute the commands:

```
cmake .
make
```

That's it! 

## Links
[Turtlebot 2 setup](Documentation/TurtlebotSetup.md)
