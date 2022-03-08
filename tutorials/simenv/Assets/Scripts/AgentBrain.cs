using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class AgentBrain : Agent
{

    private JointController _jointController;
    private GameObject _anchor;
    private GameObject _endEffector;

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
        _jointController.ResetJoints();
    }

    // Observations are collected that we want to send back to the Python side
    // Here: joint angles and joint positions in the space, and the position of the end effector (= last game object within manipulator).
    public override void CollectObservations(VectorSensor sensor)
    {
        // Foreach joint -> current angle, position (3d)
        foreach (var articulationBody in _jointController.ArticulationBodies)
        {
            sensor.AddObservation(articulationBody.xDrive.target);
            
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
}
