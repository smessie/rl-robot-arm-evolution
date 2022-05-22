
## Docs

- [Python Docs](https://smessie.github.io/SELab3-2022-01/python/)
- [Unity Project Docs](https://smessie.github.io/SELab3-2022-01/unity/)

## Config

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
  - `gamma`: 0.99
  - `eps_end`: 0.2
  - `eps_decay`: 0.999995
  - `batch_size`: 64
  - `mem_size`: 1000
  - `eps_start`: 1 
  - `hidden_nodes`: 64
  - `workspace_discretization`: 0.2
  - `goal_bal_diameter`: 0.6


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

