using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class AgentBrain : Agent
{

    private JointController _jointController;
    private GameObject _anchor;
    private GameObject _endEffector;

    private bool _makeScreenshots = false;
    private int _screenshotCounter = 0;

    // Start function of Agent class, will be called before something else happens.
    private void Awake()
    {
        _jointController = GetComponentInParent<JointController>();
        _anchor = GetComponentInParent<Builder>().anchor;
        _endEffector = GetComponentInParent<Builder>().EndEffector;
    }

    // Always called first, where manipulator is reset to begin state.
    // Here: all joint angles back to zero.
    public override void OnEpisodeBegin()
    {
        _jointController.RandomizeJoints();
    }

    // Observations are collected that we want to send back to the Python side
    // Here: joint angles and joint positions in the space, and the position of the end effector (= last game object within manipulator).
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
        int numObsToAdd = 43 - 4 * _jointController.ArticulationBodies.Count - 3;
        for (int i = 0; i < numObsToAdd; i++)
        {
            sensor.AddObservation(0f);
        }
    }

    // We get an action (within action buffer), and we'll apply this action via joint controller on the joints.
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
        if (_makeScreenshots && _screenshotCounter % 100 == 0) {
            ScreenCapture.CaptureScreenshot("screenshot-" + _screenshotCounter / 100 + ".png", 20);
        }
    }
}
