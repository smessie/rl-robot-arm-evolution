using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq.Expressions;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class WallSC : SideChannel
{
    private WallBuilder _wallBuilder;

    public WallSC(GameObject manipulator)
    {
        _wallBuilder = manipulator.GetComponent<WallBuilder>();

        ChannelId = new Guid("428c60cd-9363-4ec1-bf5e-489ec58756f1");
    }

    private int invocation = 0;

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // Format of goalPosString is (x, y, z)
        // string wall = msg.ReadString();

        bool[,] wall = new bool[13,13];
        for (int r = 0; r < 13; r++) {
            for (int c = 0; c < 13; c++) {
                wall[r,c] = c > invocation;
            }
        }
        invocation++;

        _wallBuilder.BuildWall(wall);

        string stringToSend = "Wall built";
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }
}
