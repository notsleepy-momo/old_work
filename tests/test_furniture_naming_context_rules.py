import unittest

from agents.data_models import FurnitureNaming
from agents.furniture_naming_agent import FurnitureNamingAgent


ROOM2_LAYOUT = {
    'house': {
        'rooms': [{
            'id': 'room2',
            'position': {'x': 10.24, 'y': 59.21, 'width': 33.53, 'height': 62.27},
            'furniture': [
                {'id': 'room2_fur1', 'position': {'x': 10.57, 'y': 91.00, 'width': 6.19, 'height': 30.14}},
                {'id': 'room2_fur2', 'position': {'x': 36.75, 'y': 101.33, 'width': 6.69, 'height': 9.00}},
                {'id': 'room2_fur3', 'position': {'x': 10.57, 'y': 70.36, 'width': 6.94, 'height': 11.89}},
                {'id': 'room2_fur4', 'position': {'x': 38.40, 'y': 115.20, 'width': 5.04, 'height': 5.95}},
                {'id': 'room2_fur5', 'position': {'x': 25.19, 'y': 100.58, 'width': 7.76, 'height': 14.78}},
                {'id': 'room2_fur6', 'position': {'x': 25.19, 'y': 93.15, 'width': 7.60, 'height': 7.43}},
                {'id': 'room2_fur7', 'position': {'x': 31.55, 'y': 66.15, 'width': 11.73, 'height': 14.20}},
                {'id': 'room2_fur8', 'position': {'x': 37.66, 'y': 61.52, 'width': 5.62, 'height': 4.62}},
            ],
        }],
    },
}


class FurnitureNamingContextRuleTests(unittest.TestCase):
    def setUp(self):
        # Avoid constructing an LLM client; the post-processing rules are
        # deterministic and can be exercised directly.
        self.agent = FurnitureNamingAgent.__new__(FurnitureNamingAgent)
        self.agent._ablation = {}

    def test_room2_seating_axis_repairs_joint_label_swap(self):
        namings = [
            FurnitureNaming(furniture_id='room2_fur1', name='sofa', room_id='room2', confidence=.95),
            FurnitureNaming(furniture_id='room2_fur2', name='TV Stand', room_id='room2', confidence=.95),
            FurnitureNaming(furniture_id='room2_fur3', name='tv_stand', room_id='room2', confidence=.78),
            FurnitureNaming(furniture_id='room2_fur4', name='chair', room_id='room2', confidence=.73),
            FurnitureNaming(furniture_id='room2_fur5', name='dining_table', room_id='room2', confidence=.82),
            FurnitureNaming(furniture_id='room2_fur6', name='coffee_table', room_id='room2', confidence=.90),
            FurnitureNaming(furniture_id='room2_fur7', name='dining_table_3', room_id='room2', confidence=.45),
            FurnitureNaming(furniture_id='room2_fur8', name='shoe_cabinet', room_id='room2', confidence=.40),
        ]

        repaired = self.agent._apply_hard_geometry_rules(
            namings, ROOM2_LAYOUT, {'room2': 'living_room'})
        names_by_id = {naming.furniture_id: naming.name for naming in repaired}

        self.assertEqual(names_by_id['room2_fur5'], 'coffee_table')
        self.assertEqual(names_by_id['room2_fur6'], 'chair')
        self.assertEqual(names_by_id['room2_fur7'], 'dining_table')
        self.assertEqual(names_by_id['room2_fur4'], 'chair_2')

    def test_context_rule_requires_an_observed_tv_device(self):
        namings = [
            FurnitureNaming(furniture_id='room2_fur1', name='sofa', room_id='room2', confidence=.95),
            FurnitureNaming(furniture_id='room2_fur2', name='tv_stand', room_id='room2', confidence=.95),
            FurnitureNaming(furniture_id='room2_fur5', name='dining_table', room_id='room2', confidence=.82),
            FurnitureNaming(furniture_id='room2_fur6', name='coffee_table', room_id='room2', confidence=.90),
            FurnitureNaming(furniture_id='room2_fur7', name='dining_table_3', room_id='room2', confidence=.45),
        ]
        geometry = {
            'room2_fur1': {'x': 10, 'y': 90, 'width': 6, 'height': 30},
            'room2_fur2': {'x': 36, 'y': 101, 'width': 7, 'height': 9},
            'room2_fur5': {'x': 25, 'y': 100, 'width': 8, 'height': 15},
            'room2_fur6': {'x': 25, 'y': 93, 'width': 8, 'height': 7},
            'room2_fur7': {'x': 31, 'y': 66, 'width': 12, 'height': 14},
        }
        for feature in geometry.values():
            feature['area'] = feature['width'] * feature['height']
            feature['ratio'] = max(feature['width'], feature['height']) / min(feature['width'], feature['height'])

        names_by_id = {naming.furniture_id: naming for naming in namings}
        self.agent._apply_living_room_context_rules(
            ROOM2_LAYOUT['house']['rooms'][0], names_by_id, geometry, {'room2': set()})

        self.assertEqual(names_by_id['room2_fur5'].name, 'dining_table')
        self.assertEqual(names_by_id['room2_fur6'].name, 'coffee_table')
        self.assertEqual(names_by_id['room2_fur7'].name, 'dining_table_3')


if __name__ == '__main__':
    unittest.main()
