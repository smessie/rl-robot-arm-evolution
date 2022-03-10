using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class SideChannelRegistrator : MonoBehaviour
{
    private CreationSC _creationSC;

    public void Awake()
    {
        _creationSC = new CreationSC(gameObject);

        SideChannelManager.RegisterSideChannel(_creationSC);
    }

    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(_creationSC);
        }
    }
}
