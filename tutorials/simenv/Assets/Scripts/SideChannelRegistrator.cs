using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class SideChannelRegistrator : MonoBehaviour
{
    public GameObject goal;
    public GameObject anchor;

    private CreationSC _creationSC;
    private GoalSC _goalSC;

    public void Awake()
    {
        _creationSC = new CreationSC(gameObject);
        _goalSC = new GoalSC(goal, anchor);

        SideChannelManager.RegisterSideChannel(_creationSC);
        SideChannelManager.RegisterSideChannel(_goalSC);
    }

    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(_creationSC);
            SideChannelManager.UnregisterSideChannel(_goalSC);
        }
    }
}
