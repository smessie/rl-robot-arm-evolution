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
        [XmlElement("rotation")] public RotationSpec RotationSpec;
    }

    public class JointSpec
    {
        [XmlElement("parent")] public JointLink ParentLink;
        [XmlElement("child")] public JointLink ChildLink;
        [XmlElement("angle")] public AngleSpec AngleSpec;
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

    public class RotationSpec
    {
        [XmlAttribute("lower_bound")] public float LowerBound;
        [XmlAttribute("upper_bound")] public float UpperBound;
    }

    public class AngleSpec
    {
        [XmlAttribute("lower_bound")] public float LowerBound;
        [XmlAttribute("upper_bound")] public float UpperBound;
    }
}