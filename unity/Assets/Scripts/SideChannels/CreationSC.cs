using System;
using Unity.MLAgents.SideChannels;
using UnityEngine;

/// <summary>
/// Side channel to create the robot when initializing
/// </summary>
public class CreationSC : SideChannel
{
    /// <summary>
    /// A class to store info about a built robot.
    /// Used to return info in JSON format through the creation side channel
    /// </summary>
    public class RobotInfo
    {
        /// <summary>
        /// A string containing a status indicating if the robot was built succesful or not
        /// </summary>
        public string Status;
        /// <summary>
        /// The amount of joints the newly built robot has
        /// </summary>
        public int JointAmount;
    }

    private ArmBuilder _manipulatorBuilder;

    /// <summary>
    /// Constructor that takes the main manipulator GameObject.
    /// <param name="manipulator">object that has the main scripts, like Builder.cs etc</param>
    /// </summary>
    public CreationSC(GameObject manipulator)
    {
        _manipulatorBuilder = manipulator.GetComponent<ArmBuilder>();

        ChannelId = new Guid("2c137891-46b7-4284-94ff-3dc14a7ab993");
    }

    /// <summary>
    /// Is called when a message is sent through the side channel.
    /// Uses the Builder to build the agent (robot arm).
    /// <param name="msg">The message, containing a string of the URDF format</param>
    /// </summary>
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
