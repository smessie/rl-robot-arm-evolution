using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class WorkspaceSC : SideChannel
{
    private GameObject _workspaceBox;
    private GameObject _anchor;

    public WorkspaceSC(GameObject workspaceBox, GameObject anchor)
    {
        _workspaceBox = workspaceBox;
        _anchor = anchor;

        ChannelId = new Guid("cf5e0f06-5f91-45b7-94a7-9ffe954f8bf9");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // Format of workspacePos is (x, y, z, sideLength)
        IList<float> workspacePos = msg.ReadFloatList();
        if (workspacePos == null || workspacePos.Count != 4) return;

        _workspaceBox.transform.position = _anchor.transform.position + new Vector3(workspacePos[0], workspacePos[1], workspacePos[2]);
        _workspaceBox.transform.localScale = new Vector3(workspacePos[3], workspacePos[3], workspacePos[3]);

        const string stringToSend = "Workspace set";
        using var msgOut = new OutgoingMessage();
        msgOut.WriteString(stringToSend);
        QueueMessageToSend(msgOut);
    }
}
