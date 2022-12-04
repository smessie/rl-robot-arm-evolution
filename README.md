# A Fully Self-Learning Robotic Arm

This project was made for the course [Software Engineering Lab 3](https://studiegids.ugent.be/2021/EN/studiefiches/C004072.pdf) at Ghent University.

This research aims to investigate the feasibility of making a system that can fully self-learn the shape and behavior of a robotic arm given a predetermined and varying environments.

## Docs

- [Python Docs](https://smessie.github.io/SELab3-2022-01/python/annotated.html)
- [Unity Project Docs](https://smessie.github.io/SELab3-2022-01/unity/annotated.html)

## Results

- [Conclusions can be found in the wiki](https://github.com/smessie/SELab3-2022-01/wiki/Final-results)


## Setup

- Create a virtual environment and enter it.
- Install the requirements:
```
pip install -r requirements.txt
```

- Install ml-agents via [these](https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs/Installation.md) instructions.
```
git clone --branch release_19 https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
```

#### Building unity

- Open the `unity/` folder in Unity 2020.3.30f1

- Go to Window > Package Manager.
  + Click '+' and 'Add package from disk...'. Open `com.unity.ml-agents/package.json` from your ml-agents clone.
  + At the top, select "In Registry"
  + Search for "Input System" and "Cinemachine" and install them

- Probably also a good idea: go to Edit > Preferences > External Tools > Regenerate project files

- Open `Scenes/` in the Project File Explorer and double click RobotArmScene

- Go to File > Build Settings...
  + Click "Add open scenes"
  + Click "Build". Build the project with name 'simenv' in a directory `build/` in the repository root.

## How to run

All following commands should be executed in the root of the project inside your virtual environment which was configured in the Setup section.

#### Coevolution
```
python src/run.py coevolution src/configs/example.yaml
```

#### Reinforcement learning
```
python src/run.py rl src/configs/example.yaml
```

#### Morphological evolution
```
python src/run.py morphevo src/configs/example.yaml
```

#### Environment tests
```
python ./src/run.py start_test_env src/configs/example.yaml
```

## Demo


https://user-images.githubusercontent.com/30521286/169860706-c2e6a811-2cd6-4289-ba29-e1c8d2fbe414.mp4


## Setting up C# code style analysis

First off, format whitespace and braces with `dotnet format unity.sln` in the `unity/` folder.

For code style, we use [Roslynator](https://github.com/JosefPihrt/Roslynator). Install it for your editor.

There is a `unity/unity.ruleset` file for specific rules for this Unity project.
When analyzing the code: add these lines to your `unity/Assembly-CSharp.csproj`:
```
  <PropertyGroup>
    <CodeAnalysisRuleSet>unity.ruleset</CodeAnalysisRuleSet>
  </PropertyGroup>
```
We also ignore errors in the `unity/Assets/StarterAssets/` folder, because that was included from the StarterAssets package.

Note: for it to work on your setup in Visual Studio Code, you might have to downgrade the Visual Studio Code C# Extension to 1.24.1. See [this issue](https://github.com/OmniSharp/omnisharp-vscode/issues/5160).


## What do the config parameters mean?

#### environment: parameters that are used for the environment and training
  - `path_to_unity_executable`: path to the unity executable
  - `path_to_robot_urdf`: path to a urdf file that represents an arm, is only for rl
  - `morphevo_use_graphics`: use graphics for morphevo sampling
  - `rl_use_graphics_training`: use graphics for rl training
  - `rl_use_graphics_testing`: use graphics for rl testing
  - `amount_of_cores`: amount of cores you want to use, only used in morphevo

#### arm: parameters that describe the arm
  - `minimum_amount_modules`: minimum amount modules excluding the anchor
  - `maximum_amount_modules`: maximum amount modules excluding the anchor
  - `length_lowerbound`: lowerbound of the length of a module
  - `length_upperbound`: upperbound of the length of a module
  - `movements`: possible movements a module can make (`complex` is both rotating and tilting): [`rotate`, `tilt`, `complex`]

#### mutation: parameters that will be used for mutation in coevolution and morphologic evolution
  - `standard_deviation_length`: standard deviation used to mutate length of module
  - `chance_module_drop`: chance one of the modules gets dropped while mutating, maximum one module will be dropped
  - `chance_module_add`: chance one module gets added while mutating, maximum one module will added dropped
  - `chance_type_mutation`: chance the type of a module mutates, all modules can mutate, chance will be ran every time

#### coevolution: parameters used in coevolution
  - `generations`: amount of generations you want to coevolve
  - `parents`: amount of parents that will produce new children
  - `children`: amount of children produced from parents
  - `crossover_children`: amount of the children that will undergo crossover

#### morphevo: parameters used in evolution
  - `generations`: amount of generations you want to run evolution on arms
  - `parents`: amount of parents that will produce new children
  - `children`: amount of children produced from parents
  - `crossover_children`: amount of the children that will undergo crossover
  - `sample_size`: amount of angles you want to sample to calculate coverage in workspace
  - `workspace_type`: type of workspace: normalized_cube or moved_cube
  - `workspace_cube_offset`: tuple containing the offset of the moved cube
  - `workspace_side_length`: the length of each side in case the workspace is a normalized or moved cube


#### rl: parameters used in rl
  - `episodes`: amount of episodes you want to run rl
  - `steps_per_episode`: amount of steps you want to do every episode
  - `gamma`: how important do we find future rewards? Higher gamma = more important.
  - `eps_end`: the lowest the epsilon value will go to. This is the value that will be reached when epsilon is fully decayed.
  - `eps_decay`: how fast should epsilon decay. Higher = faster.
  - `batch_size`: size of batch that is sampled from replay memory.
  - `mem_size`: size of the replay memory.
  - `eps_start`: the first value of epsilon, before there was any decay.
  - `hidden_nodes`: the size of the middle layers of the DQN model.
  - `goal_bal_diameter`: the diameter of the goal ball. This is essentially the distance from the goal center the end effector has to be at to be seen as "goal reached".
  - `use_walls`: boolean that decides whether or not to use the randomnly chosen walls during training. If this parameter is not present in the config file, it is assumed to be False.
