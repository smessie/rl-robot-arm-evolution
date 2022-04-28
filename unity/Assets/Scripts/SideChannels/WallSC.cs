using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class WallSC : SideChannel
{
    private WallBuilder _wallBuilder;

    [System.Serializable]
    class WallSpec {
        public List<WallRow> wall;
    }
    [System.Serializable]
    class WallRow {
        public List<bool> row;
    }

    public WallSC(GameObject manipulator)
    {
        _wallBuilder = manipulator.GetComponent<WallBuilder>();

        ChannelId = new Guid("428c60cd-9363-4ec1-bf5e-489ec58756f1");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        string wallString = msg.ReadString();
        WallSpec wallSpec = JsonUtility.FromJson<WallSpec>(wallString);

        List<List<bool>> wall = new List<List<bool>>();
        foreach (WallRow wallRow in wallSpec.wall) {
            wall.Add(wallRow.row);
        }

        _wallBuilder.BuildWall(wall);

        string stringToSend = "Wall built";
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }
}
