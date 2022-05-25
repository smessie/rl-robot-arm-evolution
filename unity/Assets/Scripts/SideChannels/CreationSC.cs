using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class CreationSC : SideChannel
{
    public class RobotInfo
    {
        public string Status;
        public int JointAmount;
    }

    private Builder _manipulatorBuilder;

    public CreationSC(GameObject manipulator)
    {
        _manipulatorBuilder = manipulator.GetComponent<Builder>();

        ChannelId = new Guid("2c137891-46b7-4284-94ff-3dc14a7ab993");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        string urdf = msg.ReadString();

        bool success = _manipulatorBuilder.BuildAgent(urdf);

        RobotInfo robotInfo = new RobotInfo
        {
            Status = success ? "success" : "failed",
            JointAmount = _manipulatorBuilder.GetJointAmount()
        };

        string stringToSend = JsonUtility.ToJson(robotInfo);
        using var msgOut = new OutgoingMessage();
        msgOut.WriteString(stringToSend);
        QueueMessageToSend(msgOut);
    }
}
