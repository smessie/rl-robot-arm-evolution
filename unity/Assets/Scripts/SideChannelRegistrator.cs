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
    private WallSC _wallSC;

    public void Awake()
    {
        _creationSC = new CreationSC(gameObject);
        _goalSC = new GoalSC(goal, anchor);
        _wallSC = new WallSC(gameObject);

        SideChannelManager.RegisterSideChannel(_creationSC);
        SideChannelManager.RegisterSideChannel(_goalSC);
        SideChannelManager.RegisterSideChannel(_wallSC);
    }

    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(_creationSC);
            SideChannelManager.UnregisterSideChannel(_goalSC);
            SideChannelManager.UnregisterSideChannel(_wallSC);
        }
    }
}
