using System.Collections.Generic;
using UnityEngine;

public class JointController : MonoBehaviour
{
    // Start is called before the first frame update
    public int stepSize = 10;

    public List<ArticulationBody> ArticulationBodies { get; set; }

    public void ActuateJoint(int jointIndex, float step)
    {
        step = Mathf.Clamp(step, -1f, 1f);

        var articulationBody = ArticulationBodies[jointIndex];

        var xDrive = articulationBody.xDrive;
        var newTarget = xDrive.target + step * step;
        xDrive.target = Mathf.Clamp(newTarget, xDrive.lowerLimit, xDrive.upperLimit);
        articulationBody.xDrive = xDrive;
    }

    public void ResetJoints()
    {
        foreach (var articulationBody in ArticulationBodies)
        {
            var xDrive = articulationBody.xDrive;
            xDrive.target = 0f;
            articulationBody.xDrive = xDrive;
        }
    }
}