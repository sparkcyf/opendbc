import unittest

from opendbc.car.structs import CarParams
from opendbc.car.subaru.values import SubaruSafetyFlags
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.safety_replay.helpers import get_steer_value, is_steering_msg


class TestSubaruAngleSafetyReplayHelpers(unittest.TestCase):
  def setUp(self):
    self.mode = CarParams.SafetyModel.subaru
    self.angle_param = SubaruSafetyFlags.GEN2 | SubaruSafetyFlags.LKAS_ANGLE
    self.packer = CANPackerSafety("subaru_global_2017_generated")

  def test_angle_steering_message_selection(self):
    self.assertTrue(is_steering_msg(self.mode, self.angle_param, 0x124))
    self.assertFalse(is_steering_msg(self.mode, self.angle_param, 0x122))
    self.assertTrue(is_steering_msg(self.mode, SubaruSafetyFlags.GEN2, 0x122))
    self.assertFalse(is_steering_msg(self.mode, SubaruSafetyFlags.GEN2, 0x124))

  def test_angle_steering_value_decode(self):
    msg = self.packer.make_can_msg_safety("ES_LKAS_ANGLE", 0, {"LKAS_Output": -123.45, "LKAS_Request": 1, "SET_3": 3})
    torque, angle = get_steer_value(self.mode, self.angle_param, msg)

    self.assertEqual(torque, 0)
    self.assertEqual(angle, -12345)


if __name__ == "__main__":
  unittest.main()
