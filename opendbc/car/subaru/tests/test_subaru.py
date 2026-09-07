import unittest

from opendbc.can import CANPacker
from opendbc.car import CanData
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


class TestSubaruAngleRejection(unittest.TestCase):
  def setUp(self):
    self.CI = CarInterface(CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023))
    self.packer = CANPacker("subaru_global_2017_generated")
    self.CI.update([])  # Register the signals before feeding CAN.

  def _update(self, enabled=True, messages=(), warning=False):
    status = self.packer.make_can_msg("ES_Status", 1, {"Cruise_Activated": enabled})
    torque = self.packer.make_can_msg("Steering_Torque", 0, {"Steer_Warning": warning})
    return self.CI.update([(0, [status, torque, *messages])])

  def test_rejection_latches_until_stock_cruise_exits(self):
    self.assertFalse(self._update().steerFaultTemporary)
    # Even a malformed rejected payload must be noticed without DBC validation.
    rejected = CanData(0x124, b"\xff" * 8, 0xC0)
    ret = self._update(messages=[rejected])
    self.assertTrue(ret.steerFaultTemporary)
    self.assertIs(ret, self.CI.CS.out)
    for _ in range(20):
      self.assertTrue(self._update(messages=[CanData(0x124, bytes(8), 0x80)]).steerFaultTemporary)
    self.assertFalse(self._update(enabled=False, messages=[rejected]).steerFaultTemporary)
    self.assertFalse(self._update().steerFaultTemporary)

  def test_only_rejected_bus_zero_angle_commands_latch(self):
    for address, bus in ((0x124, 0), (0x124, 2), (0x124, 0x80), (0x124, 0xC1), (0x221, 0xC0)):
      with self.subTest(address=address, bus=bus):
        self.assertFalse(self._update(messages=[CanData(address, bytes(8), bus)]).steerFaultTemporary)

  def test_cruise_exit_preserves_eps_warning(self):
    self._update(messages=[CanData(0x124, bytes(8), 0xC0)])
    self.assertTrue(self._update(enabled=False, warning=True).steerFaultTemporary)
    self.assertFalse(self._update(enabled=False).steerFaultTemporary)

  def test_torque_platform_ignores_angle_rejection(self):
    self.CI = CarInterface(CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK))
    self.CI.update([])
    self.assertFalse(self._update(messages=[CanData(0x124, bytes(8), 0xC0)]).steerFaultTemporary)
