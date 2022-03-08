import xml.etree.ElementTree as ET

class URDFGenerator:
    def __init__(self, genome_id:str ) -> None:
        self.urdf = ET.Element('robot')
        self.urdf.set('name', f'robot_{genome_id}')

        self._add_anchor()
        self.module_index = 0

    def _add_anchor(self) -> None:
        link = ET.SubElement(self.urdf, 'link', {'name': 'anchor'})
        visual = ET.SubElement(link, 'visual')
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'anchor_module')

    def add_module(self, length: float) -> None:
        link = ET.SubElement(self.urdf, 'link', {'name': f'module_{self.module_index}'})
        visual = ET.SubElement(link, 'visual')
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'base_module', {'length': str(length)})


        joint = ET.SubElement(self.urdf, 'joint', {'name': f'module_{self.module_index}_joint',
                                                   'type': 'revolute'})

        parent_link = f'module_{self.module_index - 1}' if self.module_index > 1 else 'anchor'

        ET.SubElement(joint, 'parent', {'link': parent_link})
        ET.SubElement(joint, 'child', {'link': f'module_{self.module_index}'})

        self.module_index += 1


    def get_urdf(self) -> str:
        return ET.tostring(self.urdf, encoding='unicode')
