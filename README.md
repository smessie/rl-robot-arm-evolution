
## Docs

- [Python Docs](https://smessie.github.io/SELab3-2022-01/python/)
- [Unity Project Docs](https://smessie.github.io/SELab3-2022-01/unity/)

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

