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
        [XmlElement("tilt")] public TiltingSpec TiltingSpec;
        [XmlElement("rotation")] public RotationSpec RotationSpec;
    }

    public class VisualSpec
    {
        [XmlElement("geometry")] public GeometrySpec Geometry;
    }

    public class GeometrySpec
    {
        [XmlElement("anchor_module")] public AnchorModule AnchorModule;
        [XmlElement("tilting_module")] public TiltingModule TiltingModule;
        [XmlElement("rotating_module")] public RotatingModule RotatingModule;
        [XmlElement("complex_module")] public ComplexModule ComplexModule;
    }

    public class AnchorModule
    {
        [XmlAttribute("length")] public float Length;
    }

    public class TiltingModule
    {
        [XmlAttribute("length")] public float Length;
    }

    public class RotatingModule
    {
        [XmlAttribute("length")] public float Length;
    }

    public class ComplexModule
    {
        [XmlAttribute("length")] public float Length;
    }

    public class TiltingSpec
    {
        [XmlAttribute("lower_bound")] public float LowerBound;
        [XmlAttribute("upper_bound")] public float UpperBound;
    }

    public class RotationSpec
    {
        [XmlAttribute("lower_bound")] public float LowerBound;
        [XmlAttribute("upper_bound")] public float UpperBound;
    }

    // Kind of unnecessary
    public class JointSpec
    {
        [XmlElement("parent")] public JointLink ParentLink;
        [XmlElement("child")] public JointLink ChildLink;
    }

    public class JointLink
    {
        [XmlAttribute("link")] public string Link;
    }
}