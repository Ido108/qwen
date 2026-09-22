import unittest
from backend.gpu_memory import GIB, reserve_for_dedicated_memory


class VideoMemoryTests(unittest.TestCase):
    def test_wddm_virtual_budget_does_not_hide_other_apps(self):
        reserved=reserve_for_dedicated_memory(22.5*GIB,17.2*GIB)
        self.assertAlmostEqual(reserved/GIB,7.3)
        self.assertAlmostEqual((22.5*GIB-reserved)/GIB,15.2)

    def test_matching_or_lagging_measurements_keep_safety_margin(self):
        self.assertEqual(reserve_for_dedicated_memory(17*GIB,17*GIB),2*GIB)
        self.assertEqual(reserve_for_dedicated_memory(17*GIB,18*GIB),2*GIB)
