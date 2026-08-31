import unittest

from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, SubaruSafetyFlags
from opendbc.car.structs import CarParams


class TestSubaruFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          assert len(fw) == fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}"


class TestSubaruAnglePlatform(unittest.TestCase):
  def test_only_outback_2023_is_enabled(self):
    outback = CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023)

    self.assertFalse(outback.dashcamOnly)
    self.assertEqual(outback.steerControlType, CarParams.SteerControlType.angle)
    self.assertEqual(outback.safetyConfigs[0].safetyParam, SubaruSafetyFlags.GEN2 | SubaruSafetyFlags.LKAS_ANGLE)
    self.assertFalse(outback.alphaLongitudinalAvailable)
    self.assertFalse(outback.openpilotLongitudinalControl)

    for candidate in (CAR.SUBARU_FORESTER_2022, CAR.SUBARU_ASCENT_2023):
      with self.subTest(candidate=candidate):
        self.assertTrue(CarInterface.get_non_essential_params(candidate).dashcamOnly)
