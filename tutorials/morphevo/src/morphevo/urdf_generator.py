import xml.etree.ElementTree as ET


class URDFGenerator:
    def __init__(self, genome_id: str) -> None:
        self.urdf = ET.Element('robot')
        self.urdf.set('name', f'robot_{genome_id}')

        self._add_anchor()
        self.module_index = 0

    def _add_anchor(self) -> None:
        link = ET.SubElement(self.urdf, 'link', {'name': 'anchor'})
        visual = ET.SubElement(link, 'visual')
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'anchor_module')

    def add_module(self, length: float, rotation_lower_bound: float = 0, rotation_upper_bound: float = 180,
                   angle_lower_bound: float = 0, angle_upper_bound: float = 90) -> None:
        link = ET.SubElement(self.urdf, 'link', {
                             'name': f'module_{self.module_index}'})
        visual = ET.SubElement(link, 'visual')
        ET.SubElement(link, 'rotation', {'lower_bound': f'{rotation_lower_bound}',
                                         'upper_bound': f'{rotation_upper_bound}'})
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'base_module', {'length': str(length)})

        joint = ET.SubElement(self.urdf, 'joint', {'name': f'module_{self.module_index}_joint',
                                                   'type': 'revolute'})

        parent_link = f'module_{self.module_index - 1}' if self.module_index > 1 else 'anchor'

        ET.SubElement(joint, 'parent', {'link': parent_link})
        ET.SubElement(joint, 'child', {'link': f'module_{self.module_index}'})
        ET.SubElement(joint, 'angle', {'lower_bound': f'{angle_lower_bound}',
                                       'upper_bound': f'{angle_upper_bound}'})

        self.module_index += 1

    def get_urdf(self) -> str:
        return ET.tostring(self.urdf, encoding='unicode')
