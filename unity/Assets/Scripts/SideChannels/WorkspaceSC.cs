using System;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

/// <summary>
/// Side channel to set the position of the workspace visualization
/// </summary>
public class WorkspaceSC : SideChannel
{
    private GameObject _workspaceBox;
    private GameObject _anchor;

    /// <summary>
    /// Constructor of this side channel
    /// <param name="workspaceBox">The GameObject that is the workspace visualization itself.
    /// The only thing that will change to this object is its position and size</param>
    /// <param name="anchor">Anchor, used to calculate the goal position relative to the anchor</param>
    /// </summary>
    public WorkspaceSC(GameObject workspaceBox, GameObject anchor)
    {
        _workspaceBox = workspaceBox;
        _anchor = anchor;

        ChannelId = new Guid("cf5e0f06-5f91-45b7-94a7-9ffe954f8bf9");
    }

    /// <summary>
    /// Is called when a message is sent through the side channel.
    /// Changes the position and size of the workspaceBox GameObject
    /// <param name="msg">The message, containing a list of floats in the format of (x, y, z, sideLength)</param>
    /// </summary>
    protected override void OnMessageReceived(IncomingMessage msg)
    {
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
