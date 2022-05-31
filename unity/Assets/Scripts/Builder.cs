using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using RobotSpecificationNamespace;
using UnityEngine;

/// <summary>
/// Script on the main manipulator to build the robot arm and remove it
/// </summary>
public class ArmBuilder : MonoBehaviour
{
    /// <summary>
    /// Each module has a type. Outside of the Invalid type, there are 4 types:
    ///
    /// Anchor: Special case for the lowest module. Has 0 or 1 joints.
    ///
    /// Tilting: Module that has 1 joint at the base that can 'tilt' the robot.
    ///
    /// Rotating: Module that has 1 joint in the middle, rotating along the axis of the module.
    ///
    /// Complex: Module that has 2 joints: a tilting and a rotating one.
    /// </summary>
    public enum ModuleType
    {
        Invalid,
        Anchor,
        Tilting,
        Rotating,
        Complex
    }

    /// <summary>
    /// Converts the XML-parsed LinkSpec to ModuleType
    /// </summary>
    public ModuleType TypeOfLink(LinkSpec link)
    {
        GeometrySpec geometry = link.VisualSpec.Geometry;
        if (geometry.AnchorModule != null) { return ModuleType.Anchor; }
        if (geometry.TiltingModule != null) { return ModuleType.Tilting; }
        if (geometry.RotatingModule != null) { return ModuleType.Rotating; }
        if (geometry.ComplexModule != null) { return ModuleType.Complex; }
        return ModuleType.Invalid;
    }

    /// <summary>Prefab instantiated by Unity</summary>
    public GameObject moduleBodyPrefab;
    /// <summary>Prefab instantiated by Unity</summary>
    public GameObject rotatingModuleBodyPrefab;
    /// <summary>Prefab instantiated by Unity</summary>
    public GameObject moduleConnectorPrefab;
    /// <summary>Prefab instantiated by Unity</summary>
    public GameObject tiltingModuleConnectorPrefab;
    /// <summary>Prefab instantiated by Unity</summary>
    public GameObject manipulatorAgentPrefab;

    /// <summary>The anchor GameObject, instantiated by Unity</summary>
    public GameObject anchor;

    private RobotSpecification robotSpecification = null;
    private List<GameObject> allRobotParts = new List<GameObject>();
    private GameObject endEffector;

    private List<ArticulationBody> _articulationBodies = new List<ArticulationBody>();

    public GameObject EndEffector => endEffector;

    private GameObject GetTailPrefab(ModuleType type)
    {
        if (type == ModuleType.Tilting || type == ModuleType.Complex)
        {
            return tiltingModuleConnectorPrefab;
        }
        return moduleConnectorPrefab;
    }

    private GameObject GetBodyPrefab(ModuleType type)
    {
        if (type == ModuleType.Tilting)
        {  // anchor, rotating and complex have rotating joint
            return moduleBodyPrefab;
        }
        return rotatingModuleBodyPrefab;
    }

    private float GetLength(GeometrySpec geometry, ModuleType type)
    {
        if (type == ModuleType.Anchor) { return geometry.AnchorModule.Length; }
        if (type == ModuleType.Tilting) { return geometry.TiltingModule.Length; }
        if (type == ModuleType.Rotating) { return geometry.RotatingModule.Length; }
        return geometry.ComplexModule.Length;
    }

    /// <summary>
    /// Build a robot arm. Should only be called once, in the beginning, because the manipulatorAgent
    /// would be instantiated twice.
    /// Saves the RobotSpec specification after parsing so the robot can be rebuild (see RebuildAgent)
    /// </summary>
    /// <param name="urdf">The robot specfication in URDF format, to be parsed by using
    /// XmlSerializer with the RobotSpec specification</param>
    /// <returns>Whether the parsing and building was successful</returns>
    public bool BuildAgent(string urdf)
    {
        // Parse URDF
        try
        {
            robotSpecification = ParseURDF(urdf);
            bool success = BuildAgent(robotSpecification);

            Instantiate(manipulatorAgentPrefab, Vector3.zero, Quaternion.identity, transform);
            return success;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Variant of BuildAgent method that takes a RobotSpec specification.
    /// Is called by BuildAgent(string) the first time and after that by RebuildAgent
    /// </summary>
    /// <param name="robotSpec">The robot specfication</param>
    /// <returns>Whether the building was successful</returns>
    private bool BuildAgent(RobotSpecification robotSpec)
    {
        try
        {
            endEffector = anchor;
            _articulationBodies = new List<ArticulationBody>();
            bool success = AddModules(robotSpec);

            GetComponent<JointController>().ArticulationBodies = _articulationBodies;

            return success;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Removes the current robot arm
    /// </summary>
    public void DestroyAgent()
    {
        foreach (var part in allRobotParts)
        {
            Destroy(part);
        }
        allRobotParts = new List<GameObject>();
    }

    /// <summary>
    /// Removes the current robot arm and rebuilds it.
    /// </summary>
    /// <returns>Whether the rebuilding was successful</returns>
    public bool RebuildAgent()
    {
        DestroyAgent();
        if (robotSpecification != null)
        {
            return BuildAgent(robotSpecification);
        }
        return false;
    }

    /// <summary>
    /// Parse a string in URDF format, to be parsed by using
    /// XmlSerializer with the RobotSpec specification
    /// </summary>
    /// <param name="urdf">The robot specfication string</param>
    /// <returns>The parsed specification</returns>
    private RobotSpecification ParseURDF(string urdf)
    {
        byte[] byteArray = Encoding.ASCII.GetBytes(urdf);
        MemoryStream stream = new MemoryStream(byteArray);

        XmlSerializer serializer = new XmlSerializer(typeof(RobotSpecification));
        return (RobotSpecification)serializer.Deserialize(stream);
    }

    /// <summary>
    /// Private method that is the core of BuildAgent(RobotSpec)
    /// </summary>
    /// <param name="robotSpec">The robot specfication</param>
    /// <returns>Whether the building was successful</returns>
    private bool AddModules(RobotSpecification robotSpec)
    {
        LinkSpec firstLink = robotSpec.Links[0];
        if (TypeOfLink(firstLink) != ModuleType.Anchor)
        {
            return false;
        }
        AddAnchorModule(firstLink);

        // Add modules
        foreach (var link in robotSpec.Links.GetRange(1, robotSpec.Links.Count - 1))
        {
            ModuleType type = TypeOfLink(link);
            if (type != ModuleType.Invalid && type != ModuleType.Anchor)
            {
                AddModule(link, type);
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Method called when building the robot to add the anchor module.
    ///
    /// Instantiates the two prefabs necessary to build this module and configures the joint if necessary
    /// </summary>
    /// <param name="anchorLink">The anchor module specification</param>
    private void AddAnchorModule(LinkSpec anchorLink)
    {
        // Add a module with the given length.
        GameObject module = new GameObject(anchorLink.Name);

        float length = anchorLink.VisualSpec.Geometry.AnchorModule.Length;
        float yPos = endEffector.transform.position.y + 0.5f;

        GameObject moduleBody = Instantiate(
            anchorLink.RotationSpec == null ? moduleBodyPrefab : rotatingModuleBodyPrefab, // type GameObject we want to make
            new Vector3(0f, yPos, 0f), // Position on where we want to instantiate it
            Quaternion.identity, // Turn/rotation
            module.transform
        );
        allRobotParts.Add(moduleBody);
        moduleBody.transform.localScale = new Vector3(1f, length * 2, 1f); // Multiply by 2 because we only show half of module

        // Change mass according to length.
        moduleBody.GetComponent<ArticulationBody>().mass = length;

        yPos += length * 2;
        GameObject moduleHead = Instantiate(
            moduleConnectorPrefab, // type GameObject we want to make
            new Vector3(0f, yPos, 0f), // Position on where we want to instantiate it
            Quaternion.identity, // Turn/rotation
            module.transform
        );
        allRobotParts.Add(moduleHead);

        moduleHead.transform.parent = moduleBody.transform;
        module.transform.parent = endEffector.transform;

        if (anchorLink.RotationSpec != null)
        {
            ConfigureRotatingJoint(moduleBody, anchorLink.RotationSpec);
        }
        endEffector = moduleHead;
    }

    /// <summary>
    /// Method called when building the robot to add a module other than the anchor module.
    ///
    /// Instantiates the three prefabs necessary to build this module and configures the joint(s)
    /// </summary>
    /// <param name="link">The module specification</param>
    /// <param name="type">The module type</param>
    private void AddModule(LinkSpec link, ModuleType type)
    {
        // Add a module with the given length.
        GameObject module = new GameObject(link.Name);

        float length = GetLength(link.VisualSpec.Geometry, type);
        float yPos = endEffector.transform.position.y + 1f;

        GameObject moduleTail = Instantiate(
            GetTailPrefab(type), // type GameObject we want to make
            new Vector3(0f, yPos, 0f), // Position on where we want to instantiate it
            Quaternion.identity, // Turn/rotation
            module.transform
        );
        allRobotParts.Add(moduleTail);

        yPos += length;
        GameObject moduleBody = Instantiate(
            GetBodyPrefab(type), // type GameObject we want to make
            new Vector3(0f, yPos, 0f), // Position on where we want to instantiate it
            Quaternion.identity, // Turn/rotation
            module.transform
        );
        allRobotParts.Add(moduleBody);
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
        allRobotParts.Add(moduleHead);

        moduleBody.transform.parent = moduleTail.transform;
        moduleHead.transform.parent = moduleBody.transform;
        module.transform.parent = endEffector.transform;

        if (type == ModuleType.Tilting || type == ModuleType.Complex)
        {
            ConfigureTiltingJoint(moduleTail, link.TiltingSpec);
        }
        if (type == ModuleType.Rotating || type == ModuleType.Complex)
        {
            ConfigureRotatingJoint(moduleBody, link.RotationSpec);
        }

        endEffector = moduleHead;
    }

    /// <summary>
    /// Method called when building the robot to configure a tilting joint
    /// </summary>
    /// <param name="moduleTail">The GameObject that 'carries' the joint</param>
    /// <param name="tiltingSpec">The specification of the joint</param>
    private void ConfigureTiltingJoint(GameObject moduleTail, TiltingSpec tiltingSpec)
    {
        ArticulationBody articulationBody = moduleTail.GetComponent<ArticulationBody>();

        articulationBody.jointType = ArticulationJointType.RevoluteJoint;
        articulationBody.anchorRotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));

        // Articulation degree of freedom
        articulationBody.twistLock = ArticulationDofLock.LimitedMotion;

        ArticulationDrive xDrive = articulationBody.xDrive;
        xDrive.lowerLimit = tiltingSpec.LowerBound;
        xDrive.upperLimit = tiltingSpec.UpperBound;
        xDrive.stiffness = 80000;
        xDrive.damping = 10000;
        articulationBody.xDrive = xDrive;

        _articulationBodies.Add(moduleTail.GetComponent<ArticulationBody>());
    }

    /// <summary>
    /// Method called when building the robot to configure a rotating joint
    /// </summary>
    /// <param name="moduleBody">The GameObject that 'carries' the joint</param>
    /// <param name="rotationSpec">The specification of the rotation</param>
    private void ConfigureRotatingJoint(GameObject moduleBody, RotationSpec rotationSpec)
    {
        ArticulationBody articulationBody = moduleBody.GetComponent<ArticulationBody>();

        articulationBody.jointType = ArticulationJointType.RevoluteJoint;
        articulationBody.anchorRotation = Quaternion.Euler(new Vector3(0f, 0f, 90f));

        // Articulation degree of freedom
        articulationBody.twistLock = ArticulationDofLock.LimitedMotion;

        ArticulationDrive xDrive = articulationBody.xDrive;

        if (rotationSpec.LowerBound < 0.0001 && rotationSpec.UpperBound > 359.9999)
        {
            xDrive.lowerLimit = -360 * 100;
            xDrive.upperLimit = 360 * 100;
            xDrive.target = 0;
        }
        else
        {
            xDrive.lowerLimit = rotationSpec.LowerBound;
            xDrive.upperLimit = rotationSpec.UpperBound;
        }

        xDrive.stiffness = 80000;
        xDrive.damping = 10000;
        articulationBody.xDrive = xDrive;

        _articulationBodies.Add(moduleBody.GetComponent<ArticulationBody>());
    }

    /// <summary>
    /// Returns the amount of joints the currently built robot arm has (zero if no arm built)
    /// </summary>
    /// <returns>The amount of joints</returns>
    public int GetJointAmount()
    {
        return _articulationBodies.Count;
    }
}
