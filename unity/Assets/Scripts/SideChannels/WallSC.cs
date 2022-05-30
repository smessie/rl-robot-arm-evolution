using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class WallSC : SideChannel
{
#pragma warning disable CS0649 // (Field is never assigned to) These fields are assigned to by JsonUtilty
    [System.Serializable]
    class WallSpec
    {
        public List<WallRow> wall;
    }
    [System.Serializable]
    class WallRow
    {
        public List<bool> row;
    }
#pragma warning restore CS0649

    private WallBuilder _wallBuilder;

    public WallSC(GameObject manipulator)
    {
        _wallBuilder = manipulator.GetComponent<WallBuilder>();

        ChannelId = new Guid("428c60cd-9363-4ec1-bf5e-489ec58756f1");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        string wallString = msg.ReadString();
        string stringToSend;
        try
        {
            WallSpec wallSpec = JsonUtility.FromJson<WallSpec>(wallString);

            List<List<bool>> wall = new List<List<bool>>();
            foreach (WallRow wallRow in wallSpec.wall)
            {
                wall.Add(wallRow.row);
            }

            _wallBuilder.BuildWall(wall);

            stringToSend = "Wall built";
        }
        catch
        {
            _wallBuilder.ClearWalls();

            stringToSend = "Walls cleared";
        }

        using var msgOut = new OutgoingMessage();
        msgOut.WriteString(stringToSend);
        QueueMessageToSend(msgOut);
    }
}
