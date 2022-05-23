
## Docs

- [Python Docs](https://smessie.github.io/SELab3-2022-01/python/)
- [Unity Project Docs](https://smessie.github.io/SELab3-2022-01/unity/)

## Config

#### arm: parameters that describe the arm
  - `minimum_amount_modules`: minimum amount modules excluding the anchor
  - `maximum_amount_modules`: maximum amount modules excluding the anchor
  - `length_lowerbound`: lowerbound of the length of a module
  - `length_upperbound`: upperbound of the length of a module
  - `movements`: possible movements a module can make: [`rotate`, `tilt`]

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
  ##### @IEBEN 
  - `workspace_type`: "moved_cube"
  - `workspace_cube_offset`: [0,5,9]
  - `workspace_side_length`: 4 


#### rl: parameters used in rl
  - `episodes`: amount of episodes you want to run rl
  - `steps_per_episode`: amount of steps you want to do every episode
  ##### @ FREYA IWIJN
  - `gamma`: how important do we find future rewards? Higher gamma = more important.
  - `eps_end`: the lowest the epsilon value will go to. This is the value that will be reached when epsilon is fully decayed.
  - `eps_decay`: how fast should epsilon decay. Higher = faster.
  - `batch_size`: size of batch that is sampled from replay memory.
  - `mem_size`: size of the replay memory.
  - `eps_start`: the first value of epsilon, before there was any decay.
  - `hidden_nodes`: the size of the middle layers of the DQN model.
  - `goal_bal_diameter`: the diameter of the goal ball. This is essentially the distance from the goal center the end effector has to be at to be seen as "goal reached".


## Building unity

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

