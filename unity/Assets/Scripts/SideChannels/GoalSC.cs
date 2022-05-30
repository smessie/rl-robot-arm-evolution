using System;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

/// <summary>
/// Side channel to set the position of the goal visualization
/// </summary>
public class GoalSC : SideChannel
{
    private GameObject _goal;
    private GameObject _anchor;

    /// <summary>
    /// Constructor of this side channel
    /// <param name="goal">The GameObject that is the goal visualization itself.
    /// The only thing that will change to this object is its position</param>
    /// <param name="anchor">Anchor, used to calculate the goal position relative to the anchor</param>
    /// </summary>
    public GoalSC(GameObject goal, GameObject anchor)
    {
        _goal = goal;
        _anchor = anchor;

        ChannelId = new Guid("2cc47842-41f8-455d-9ff7-5925d152133a");
    }

    /// <summary>
    /// Is called when a message is sent through the side channel.
    /// Changes the position of the goal GameObject
    /// <param name="msg">The message, containing a list of floats indicating the new coordinate of the goal</param>
    /// </summary>
    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // Format of goalPos is (x, y, z)
        IList<float> goalPos = msg.ReadFloatList();
        if (goalPos == null || goalPos.Count != 3) return;

        _goal.transform.position = _anchor.transform.position + new Vector3(goalPos[0], goalPos[1], goalPos[2]);

        const string stringToSend = "Goal set";
        using var msgOut = new OutgoingMessage();
        msgOut.WriteString(stringToSend);
        QueueMessageToSend(msgOut);
    }
}
