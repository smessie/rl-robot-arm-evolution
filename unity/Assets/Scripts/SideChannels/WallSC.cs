using System;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

/// <summary>
/// Side channel to place walls in the environment.
///
/// It supports two operations: adding one wall and removing all walls.
/// </summary>
public class WallSC : SideChannel
{
#pragma warning disable CS0649 // (Field is never assigned to) These fields are assigned to by JsonUtilty
    /// <summary>
    /// A serializable class used to parse JSON sent through the wall side channel
    ///
    /// Essentially represents a 2D array because the only field is a List of WallRow,
    /// which are Lists of booleans
    /// </summary>
    [System.Serializable]
    public class WallSpec
    {
        /// <summary>
        /// List of WallRow, the horizontal rows of a wall
        /// </summary>
        public List<WallRow> wall;
    }
    /// <summary>
    /// A serializable class used to parse JSON sent through the wall side channel
    ///
    /// Represents a row of tiles in the wall
    /// </summary>
    [System.Serializable]
    public class WallRow
    {
        /// <summary>
        /// List of bool, each representing a tile in the wall
        /// </summary>
        public List<bool> row;
    }
#pragma warning restore CS0649

    private WallBuilder _wallBuilder;

    /// <summary>
    /// Constructor of this side channel
    /// <param name="manipulator">object that has the main scripts, like Builder.cs etc</param>
    /// </summary>
    public WallSC(GameObject manipulator)
    {
        _wallBuilder = manipulator.GetComponent<WallBuilder>();

        ChannelId = new Guid("428c60cd-9363-4ec1-bf5e-489ec58756f1");
    }

    /// <summary>
    /// Is called when a message is sent through the side channel.
    /// When the message is a wall, instructs the WallBuilder to build a new wall.
    /// Otherwise all walls are cleared.
    /// <param name="msg">The message, either a JSON string parseable to WallSpec (for building a new wall)
    /// or any other string (to remove all walls)</param>
    /// </summary>
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
