using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// The brain of the AI agent. It has lifecycle methods,
/// receives actions and sends observations.
/// Inherits from [Unity.MLAgents.Agent]
/// </summary>
public class AgentBrain : Agent
{
    private JointController _jointController;

    private GameObject _anchor;
    private GameObject _endEffector;

    private readonly bool _makeScreenshots = true;

    private int _screenshotCounter = 0;

    // Start function of Agent class, will be called before something else happens.
    private void Awake()
    {
        _jointController = GetComponentInParent<JointController>();
        _anchor = GetComponentInParent<Builder>().anchor;
        _endEffector = GetComponentInParent<Builder>().EndEffector;
    }

    /// <summary>
    /// Lifecycle method that gets called in the beginning of an episode (by the mlagents code).
    /// The robot is rebuilt and the joints are randomized.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        GetComponentInParent<Builder>().RebuildAgent();
        _endEffector = GetComponentInParent<Builder>().EndEffector;
        _jointController.RandomizeJoints();
    }

    /// <summary>
    /// Observations are collected that we want to send back to the Python side
    /// Joint angles and joint positions in the space, and the position of the end effector (= last game object within manipulator)
    /// Like this: [j1_x, j1_y, j1_z, j1_θ, j2_x, j2_y, j2_z, j2_θ, ..., ee_x, ee_y, ee_z]
    /// </summary>
    /// <param name="sensor">Object on which observations are added</param>
    /// <returns>Nothing</returns>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Foreach joint -> current angle, position (3d)
        foreach (var articulationBody in _jointController.ArticulationBodies)
        {
            // For joints that rotate further
            float target = articulationBody.xDrive.target % 360;
            sensor.AddObservation(target >= 0 ? target : target + 360);

            sensor.AddObservation(articulationBody.transform.position - _anchor.transform.position);
        }

        // End effector -> position
        sensor.AddObservation(_endEffector.transform.position - _anchor.transform.position);

        // Make sure we will always send 43 observations.
        int numObsToAdd = 43 - (4 * _jointController.ArticulationBodies.Count) - 3;
        for (int i = 0; i < numObsToAdd; i++)
        {
            sensor.AddObservation(0f);
        }
    }

    /// <summary>
    /// Called when actions are received from the Python side
    /// </summary>
    /// <param name="actions">Object that has the actions. Every action represent how much a joint should move</param>
    /// <returns>Nothing</returns>
    public override void OnActionReceived(ActionBuffers actions)
    {
        for (int i = 0; i < _jointController.ArticulationBodies.Count; i++)
        {
            float angleStep = actions.ContinuousActions[i];
            _jointController.ActuateJoint(i, angleStep);
        }
        base.OnActionReceived(actions);
    }

    // This function is called every time before we'll do a physics update. Before all forces and positions are calculated again in the scene, this function is called.
    // All physics related updates, you perform in the FixedUpdate method.
    private void FixedUpdate()
    {
        _screenshotCounter++;
        if (_makeScreenshots && _screenshotCounter % 1000 == 0)
        {
            ScreenCapture.CaptureScreenshot("screenshot-" + (_screenshotCounter / 1000) + ".png", 15);
        }
    }
}
