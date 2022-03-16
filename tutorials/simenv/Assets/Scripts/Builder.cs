using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using DefaultNamespace;
using UnityEngine;

public class Builder : MonoBehaviour
{

    public float[] moduleLengths;

    public GameObject moduleBodyPrefab;
    public GameObject moduleConnectorPrefab;
    public GameObject manipulatorAgentPrefab;

    public GameObject anchor;

    private GameObject endEffector;

    private List<ArticulationBody> _articulationBodies = new List<ArticulationBody>();

    public GameObject EndEffector => endEffector;

    private float[] ParseURDF(string urdf)
    {
        byte[] byteArray = Encoding.ASCII.GetBytes(urdf);
        MemoryStream stream = new MemoryStream(byteArray);

        XmlSerializer serializer = new XmlSerializer(typeof(RobotSpecification));
        RobotSpecification robotSpec = (RobotSpecification) serializer.Deserialize(stream);

        // Now we extract module length, later we'll have to also extract joint angles and degrees of freedom of joint.
        List<float> ml = new List<float>();
        foreach (var link in robotSpec.Links)
        {
            BaseModule baseModule = link.VisualSpec.Geometry.BaseModule;
            if (baseModule != null)
            {
                ml.Add(baseModule.Length);
            }
        }

        return ml.ToArray();
    }

    public void BuildAgent(string urdf)
    {
        // Parse URDF
        moduleLengths = ParseURDF(urdf);

        endEffector = anchor;
        foreach (var length in moduleLengths)
        {
            AddModule(length);
        }

        GetComponent<JointController>().ArticulationBodies = _articulationBodies;

        Instantiate(manipulatorAgentPrefab, Vector3.zero, Quaternion.identity, transform);
    }

    void AddModule(float length) {
        // Add a module with the given length.
        GameObject module = new GameObject("Module");

        float yPos = endEffector.transform.position.y + 1f;

        GameObject moduleTail = Instantiate(
            moduleConnectorPrefab, // type GameObject we want to make
            new Vector3(0f, yPos, 0f), // Position on where we want to instantiate it
            Quaternion.identity, // Turn/rotation
            module.transform
        );

        yPos += length;
        GameObject moduleBody = Instantiate(
            moduleBodyPrefab, // type GameObject we want to make
            new Vector3(0f, yPos, 0f), // Position on where we want to instantiate it
            Quaternion.identity, // Turn/rotation
            module.transform
        );
        moduleBody.transform.localScale = new Vector3(1f, length, 1f);

        // Change mass according to length.
        moduleBody.GetComponent<ArticulationBody>().mass = length;

        yPos += length;
        GameObject moduleHead = Instantiate(
            moduleConnectorPrefab, // type GameObject we want to make
            new Vector3(0f, yPos, 0f), // Position on where we want to instantiate it
            Quaternion.identity, // Turn/rotation
            module.transform
        );

        moduleBody.transform.parent = moduleTail.transform;
        moduleHead.transform.parent = moduleBody.transform;
        module.transform.parent = endEffector.transform;

        ConfigureTiltingJoint(moduleTail);
        ConfigureRotatingJoint(moduleBody);

        endEffector = moduleHead;
    }

    void ConfigureTiltingJoint(GameObject moduleTail)
    {
        ArticulationBody articulationBody = moduleTail.GetComponent<ArticulationBody>();

        articulationBody.jointType = ArticulationJointType.RevoluteJoint;
        articulationBody.anchorRotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));

        // Articulation degree of freedom
        articulationBody.twistLock = ArticulationDofLock.LimitedMotion;

        ArticulationDrive xDrive = articulationBody.xDrive;
        xDrive.lowerLimit = 0f;
        xDrive.upperLimit = 90f;
        xDrive.stiffness = 100000;
        xDrive.damping = 10000;
        articulationBody.xDrive = xDrive;

        _articulationBodies.Add(moduleTail.GetComponent<ArticulationBody>());
    }

    void ConfigureRotatingJoint(GameObject moduleBody)
    {
        ArticulationBody articulationBody = moduleBody.GetComponent<ArticulationBody>();

        articulationBody.jointType = ArticulationJointType.RevoluteJoint;
        articulationBody.anchorRotation = Quaternion.Euler(new Vector3(0f, 0f, 90f));

        // Articulation degree of freedom
        articulationBody.twistLock = ArticulationDofLock.LimitedMotion;

        ArticulationDrive xDrive = articulationBody.xDrive;
        xDrive.lowerLimit = 0f;
        xDrive.upperLimit = 360f;
        xDrive.stiffness = 100000;
        xDrive.damping = 10000;
        articulationBody.xDrive = xDrive;

        _articulationBodies.Add(moduleBody.GetComponent<ArticulationBody>());
    }

    // This function is called every time before we'll do a physics update. Before all forces and positions are calculated again in the scene, this function is called.
    // All physics related updates, you perform in the FixedUpdate method.
    private void FixedUpdate()
    {
    }
}
