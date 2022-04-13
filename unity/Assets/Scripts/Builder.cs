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

    private RobotSpecification ParseURDF(string urdf)
    {
        byte[] byteArray = Encoding.ASCII.GetBytes(urdf);
        MemoryStream stream = new MemoryStream(byteArray);

        XmlSerializer serializer = new XmlSerializer(typeof(RobotSpecification));
        return (RobotSpecification) serializer.Deserialize(stream);
    }

    private void AddModules(RobotSpecification robotSpec)
    {
        LinkSpec anchorLink = robotSpec.Links[0];
        if (anchorLink.VisualSpec.Geometry.AnchorModule != null) {
            AddAnchorModule(anchorLink.VisualSpec.Geometry.AnchorModule.Length,
                            anchorLink.RotationSpec.LowerBound, anchorLink.RotationSpec.UpperBound);
        } else {
            return;
        }
        // Get joint angle bounds
        List<float> joint_angles_lower_bounds = new List<float>();
        List<float> joint_angles_upper_bounds = new List<float>();
        foreach (var joint in robotSpec.Joints)
        {
            AngleSpec angleSpec = joint.AngleSpec;
            if (angleSpec != null)
            {
                joint_angles_lower_bounds.Add(angleSpec.LowerBound);
                joint_angles_upper_bounds.Add(angleSpec.UpperBound);
            }
        }
        // Add modules
        int i = 0;
        foreach (var link in robotSpec.Links)
        {
            BaseModule baseModule = link.VisualSpec.Geometry.BaseModule;
            RotationSpec rotationSpec = link.RotationSpec;
            if (baseModule != null && rotationSpec != null)
            {
                AddModule(baseModule.Length, rotationSpec.LowerBound, rotationSpec.UpperBound,
                          joint_angles_lower_bounds[i], joint_angles_upper_bounds[i]);
                i++;
            }
        }

    }

    public void BuildAgent(string urdf)
    {
        // Parse URDF
        RobotSpecification robotSpec = ParseURDF(urdf);

        endEffector = anchor;

        AddModules(robotSpec);

        GetComponent<JointController>().ArticulationBodies = _articulationBodies;

        Instantiate(manipulatorAgentPrefab, Vector3.zero, Quaternion.identity, transform);
    }

    void AddAnchorModule(float length, float rotation_lower_bound, float rotation_upper_bound) {
        // Add a module with the given length.
        GameObject module = new GameObject("Module");

        float yPos = endEffector.transform.position.y + 0.5f + length;

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

        moduleHead.transform.parent = moduleBody.transform;
        module.transform.parent = endEffector.transform;

        ConfigureRotatingJoint(moduleBody, rotation_lower_bound, rotation_upper_bound);

        endEffector = moduleHead;
    }

    void AddModule(float length, float rotation_lower_bound, float rotation_upper_bound,
                   float angle_lower_bound, float angle_upper_bound) {
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

        ConfigureTiltingJoint(moduleTail, angle_lower_bound, angle_upper_bound);
        ConfigureRotatingJoint(moduleBody, rotation_lower_bound, rotation_upper_bound);

        endEffector = moduleHead;
    }

    void ConfigureTiltingJoint(GameObject moduleTail, float lower_bound, float upper_bound)
    {
        ArticulationBody articulationBody = moduleTail.GetComponent<ArticulationBody>();

        articulationBody.jointType = ArticulationJointType.RevoluteJoint;
        articulationBody.anchorRotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));

        // Articulation degree of freedom
        articulationBody.twistLock = ArticulationDofLock.LimitedMotion;

        ArticulationDrive xDrive = articulationBody.xDrive;
        xDrive.lowerLimit = lower_bound;
        xDrive.upperLimit = upper_bound;
        xDrive.stiffness = 100000;
        xDrive.damping = 10000;
        articulationBody.xDrive = xDrive;

        _articulationBodies.Add(moduleTail.GetComponent<ArticulationBody>());
    }

    void ConfigureRotatingJoint(GameObject moduleBody, float lower_bound, float upper_bound)
    {
        ArticulationBody articulationBody = moduleBody.GetComponent<ArticulationBody>();

        articulationBody.jointType = ArticulationJointType.RevoluteJoint;
        articulationBody.anchorRotation = Quaternion.Euler(new Vector3(0f, 0f, 90f));

        // Articulation degree of freedom
        articulationBody.twistLock = ArticulationDofLock.LimitedMotion;

        ArticulationDrive xDrive = articulationBody.xDrive;
        xDrive.lowerLimit = lower_bound;
        xDrive.upperLimit = upper_bound;
        xDrive.stiffness = 100000;
        xDrive.damping = 10000;
        articulationBody.xDrive = xDrive;

        _articulationBodies.Add(moduleBody.GetComponent<ArticulationBody>());
    }

    public int getJointAmount() {
        return _articulationBodies.Count;
    }

    // This function is called every time before we'll do a physics update. Before all forces and positions are calculated again in the scene, this function is called.
    // All physics related updates, you perform in the FixedUpdate method.
    private void FixedUpdate()
    {
    }
}
