using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class GoalSC : SideChannel
{
    private GameObject _goal;
    private GameObject _anchor;

    public GoalSC(GameObject goal, GameObject anchor)
    {
        _goal = goal;
        _anchor = anchor;

        ChannelId = new Guid("2cc47842-41f8-455d-9ff7-5925d152133a");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // Format of goalPos is (x, y, z)
        IList<float> goalPos = msg.ReadFloatList();
        if (goalPos == null || goalPos.Count != 3) return;

        _goal.transform.position = _anchor.transform.position + new Vector3(goalPos[0], goalPos[1], goalPos[2]);

        string stringToSend = "Goal set";
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }
}
