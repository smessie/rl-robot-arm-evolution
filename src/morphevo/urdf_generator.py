import sys
import xml.etree.ElementTree as ET
from typing import Tuple


class URDFGenerator:
    """! A class that helps to generate the urdf string of an arm.
    """
    def __init__(self, genome_id: str) -> None:
        """!
        @param genome_id: The id of the genome that will be translated to urdf.
        """
        self.urdf = ET.Element('robot')
        self.urdf.set('name', f'robot_{genome_id}')

        self.anchor_added = False
        self.module_index = 0

    def add_anchor(self, length: float = 1.0, can_rotate: bool = True,
                   rotation_bounds: Tuple[float] = (0, 360)) -> None:
        """! Add an anchor module to the urdf.
        @param length The length of the anchor.
        @param can_rotate Represents if an anchor can rotate.
        @param rotation_bounds The bounds in which the arm can rotate.
        """
        link = ET.SubElement(self.urdf, 'link', {'name': 'anchor'})
        visual = ET.SubElement(link, 'visual')
        if can_rotate:
            ET.SubElement(link, 'rotation', {'lower_bound': f'{rotation_bounds[0]}',
                                             'upper_bound': f'{rotation_bounds[1]}'})
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'anchor_module', {'length': str(length)})
        self.anchor_added = True

    def _get_module_name(self, can_tilt: bool, can_rotate: bool):
        """! Get the name of a module type.
        @param can_tilt If module can tilt.
        @param can_rotate If a module can rotate.
        @returns name of module type.
        """
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
        """! Add a module to the urdf string
        @param length The length of the module.
        @param can_tilt If a module can tilt.
        @param can_rotate If a module can rotate.
        @param tilt_bounds Angle bounds in between a module can tilt.
        @param rotation_bounds Angle bounds in between a module can rotate.
        """
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
        """! Get the generated urdf string.
        @returns The generated urdf string.
        """
        return ET.tostring(self.urdf, encoding='unicode')
