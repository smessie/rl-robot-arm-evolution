using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Class that keeps a list of the joints of the robot arm and can control them
/// </summary>
public class JointController : MonoBehaviour
{
    /// <summary>
    /// Actions are between -1 and 1.
    /// This variable is multiplied with the action and that is how many degrees the joint wil actuate
    /// </summary>
    public int stepSize = 10;

    /// <summary>
    /// List of joints of the robot arm
    /// </summary>
    public List<ArticulationBody> ArticulationBodies { get; set; }

    private void Awake()
    {
        bool seedProvided = int.TryParse(Environment.GetEnvironmentVariable("SEED"), out int seed);
        UnityEngine.Random.InitState(seedProvided ? seed : DateTime.Now.Millisecond);
    }

    /// <summary>
    /// Actuate one joint of the robot arm
    /// </summary>
    /// <param name="jointIndex">Which joint should be actuated. Starts at the base of the robot arm</param>
    /// <param name="step">Value that is clamped between -1 and 1, indicating which direction
    /// the actuation should be in and how big it should be. It is multiplied by the stepSize member,
    /// so the maximum actuation is stepSize degrees.</param>
    public void ActuateJoint(int jointIndex, float step)
    {
        step = Mathf.Clamp(step, -1f, 1f);

        var articulationBody = ArticulationBodies[jointIndex];

        var xDrive = articulationBody.xDrive;
        var newTarget = xDrive.target + (stepSize * step);
        if (xDrive.lowerLimit < 0.0001 && xDrive.upperLimit > 359.9999)
        {
            xDrive.target = newTarget;
        }
        else
        {
            xDrive.target = Mathf.Clamp(newTarget, xDrive.lowerLimit, xDrive.upperLimit);
        }

        articulationBody.xDrive = xDrive;
    }

    /// <summary>
    /// Set all the targets of the robot joints to 0.
    /// </summary>
    public void ResetJoints()
    {
        foreach (var articulationBody in ArticulationBodies)
        {
            var xDrive = articulationBody.xDrive;
            xDrive.target = 0f;
            articulationBody.xDrive = xDrive;
        }
    }

    /// <summary>
    /// Set all the targets of the robot joints to a random amount of degrees between -20 and 20,
    /// clamped by the limits of the joints.
    /// </summary>
    public void RandomizeJoints()
    {
        foreach (var articulationBody in ArticulationBodies)
        {
            var xDrive = articulationBody.xDrive;
            xDrive.target = UnityEngine.Random.Range(Mathf.Max(-20, xDrive.lowerLimit), Mathf.Min(20, xDrive.upperLimit));
            articulationBody.xDrive = xDrive;
        }
    }
}