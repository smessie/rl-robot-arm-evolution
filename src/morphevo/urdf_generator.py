import sys
import xml.etree.ElementTree as ET
from typing import Tuple


class URDFGenerator:
    def __init__(self, genome_id: str) -> None:
        self.urdf = ET.Element('robot')
        self.urdf.set('name', f'robot_{genome_id}')

        self.anchor_added = False
        self.module_index = 0

    def add_anchor(self, length: float = 0.5, can_rotate: bool = False,
                   rotation_bounds: Tuple[float] = (0, 180)) -> None:
        link = ET.SubElement(self.urdf, 'link', {'name': 'anchor'})
        visual = ET.SubElement(link, 'visual')
        if can_rotate:
            ET.SubElement(link, 'rotation', {'lower_bound': f'{rotation_bounds[0]}',
                                             'upper_bound': f'{rotation_bounds[1]}'})
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'anchor_module', {'length': str(length)})
        self.anchor_added = True

    def _get_module_name(self, can_tilt: bool, can_rotate: bool):
        if can_tilt:
            if can_rotate:
                return 'complex_module'
            return 'tilting_module'
        if can_rotate:
            return 'rotating_module'
        print("Module should either rotate or tilt")
        sys.exit(0)

    def add_module(self, length: float, can_tilt: bool = True, can_rotate: bool = False,
                   tilt_bounds: Tuple[float] = (0, 90),
                   rotation_bounds: Tuple[float] = (0, 180)) -> None:
        if not self.anchor_added:
            raise Exception("Anchor has to be added first")

        link = ET.SubElement(self.urdf, 'link', {
                             'name': f'module_{self.module_index}'})
        visual = ET.SubElement(link, 'visual')
        geometry = ET.SubElement(visual, 'geometry')
        module_name = self._get_module_name(can_tilt, can_rotate)
        ET.SubElement(geometry, module_name, {'length': str(length)})

        if can_tilt:
            ET.SubElement(link, 'tilt', {'lower_bound': f'{tilt_bounds[0]}',
                                        'upper_bound': f'{tilt_bounds[1]}'})
        if can_rotate:
            ET.SubElement(link, 'rotation', {'lower_bound': f'{rotation_bounds[0]}',
                                             'upper_bound': f'{rotation_bounds[1]}'})


        joint = ET.SubElement(self.urdf, 'joint', {'name': f'module_{self.module_index}_joint',
                                                   'type': 'revolute'})

        parent_link = f'module_{self.module_index - 1}' if self.module_index > 1 else 'anchor'

        ET.SubElement(joint, 'parent', {'link': parent_link})
        ET.SubElement(joint, 'child', {'link': f'module_{self.module_index}'})
        self.module_index += 1

    def get_urdf(self) -> str:
        return ET.tostring(self.urdf, encoding='unicode')
