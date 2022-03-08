using System.Collections.Generic;
using System.Xml.Serialization;

namespace DefaultNamespace
{
    [XmlRoot("robot")]
    public class RobotSpecification
    {
        [XmlAttribute("name")] public string Name;

        [XmlElement("link")] public List<LinkSpec> Links = new List<LinkSpec>();

        [XmlElement("joint")] public List<JointSpec> Joints = new List<JointSpec>();
    }

    public class LinkSpec
    {
        [XmlAttribute("name")] public string Name;
        [XmlElement("visual")] public VisualSpec VisualSpec;
    }
    
    public class JointSpec
    {
        [XmlElement("parent")] public JointLink ParentLink;
        [XmlElement("child")] public JointLink ChildLink;
    }

    public class JointLink
    {
        [XmlAttribute("link")] public string Link;
    }

    public class VisualSpec
    {
        [XmlElement("geometry")] public GeometrySpec Geometry;
    }

    public class GeometrySpec
    {
        [XmlElement("anchor_module")] public AnchorModule AnchorModule;
        [XmlElement("base_module")] public BaseModule BaseModule;
    }

    public class BaseModule
    {
        [XmlAttribute("length")] public float Length;
    }

    public class AnchorModule
    {
    }
}