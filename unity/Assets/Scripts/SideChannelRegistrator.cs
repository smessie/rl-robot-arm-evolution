using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class SideChannelRegistrator : MonoBehaviour
{
    public GameObject anchor;
    public GameObject goal;
    public GameObject workspaceBox;

    private CreationSC _creationSC;
    private GoalSC _goalSC;
    private WorkspaceSC _workspaceSC;
    private WallSC _wallSC;

    public void Awake()
    {
        _creationSC = new CreationSC(gameObject);
        _goalSC = new GoalSC(goal, anchor);
        _workspaceSC = new WorkspaceSC(workspaceBox, anchor);
        _wallSC = new WallSC(gameObject);

        SideChannelManager.RegisterSideChannel(_creationSC);
        SideChannelManager.RegisterSideChannel(_goalSC);
        SideChannelManager.RegisterSideChannel(_workspaceSC);
        SideChannelManager.RegisterSideChannel(_wallSC);
    }

    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(_creationSC);
            SideChannelManager.UnregisterSideChannel(_goalSC);
            SideChannelManager.UnregisterSideChannel(_workspaceSC);
            SideChannelManager.UnregisterSideChannel(_wallSC);
        }
    }
}
