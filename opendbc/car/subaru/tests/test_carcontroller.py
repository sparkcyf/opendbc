import unittest
from types import SimpleNamespace

from opendbc.car.lateral import apply_steer_angle_limits_vm
from opendbc.car.subaru.carcontroller import CarController
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, CarControllerParams
from opendbc.car.vehicle_model import calc_slip_factor


class TestSubaruCarController(unittest.TestCase):
  @staticmethod
  def _controller():
    CP = CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023)
    return CarController({}, CP)

  @staticmethod
  def _state(speed, steering_angle):
    return SimpleNamespace(out=SimpleNamespace(vEgoRaw=speed, steeringAngleDeg=steering_angle))

  @staticmethod
  def _control(lat_active, requested_angle):
    return SimpleNamespace(latActive=lat_active, actuators=SimpleNamespace(steeringAngleDeg=requested_angle))

  def test_controller_vehicle_model_matches_panda_safety(self):
    controller = self._controller()
    assert controller.VM is not None

    self.assertAlmostEqual(controller.VM.l, 2.89, places=6)
    self.assertAlmostEqual(controller.VM.sR, 13.5, places=6)
    self.assertAlmostEqual(calc_slip_factor(controller.VM), -0.000580374471400815)

  def test_lkas_angle_rising_edge_uses_live_steering_angle(self):
    controller = self._controller()
    controller.apply_angle_last = 2.46

    CS = self._state(15.0, 2.61)
    CC = self._control(True, 1.11)
    controller.handle_angle_lateral(CC, CS)

    expected = apply_steer_angle_limits_vm(CC.actuators.steeringAngleDeg, CS.out.steeringAngleDeg, CS.out.vEgoRaw,
                                           CS.out.steeringAngleDeg, CC.latActive, CarControllerParams, controller.VM)
    stale_reference = apply_steer_angle_limits_vm(CC.actuators.steeringAngleDeg, 2.46, CS.out.vEgoRaw,
                                                  CS.out.steeringAngleDeg, CC.latActive, CarControllerParams, controller.VM)

    self.assertAlmostEqual(controller.apply_angle_last, expected)
    self.assertNotAlmostEqual(controller.apply_angle_last, stale_reference)

  def test_lkas_angle_does_not_reanchor_while_active(self):
    controller = self._controller()
    controller.apply_angle_last = 2.46
    controller.lat_active_prev = True

    CS = self._state(15.0, 20.0)
    CC = self._control(True, 1.11)
    controller.handle_angle_lateral(CC, CS)

    expected = apply_steer_angle_limits_vm(CC.actuators.steeringAngleDeg, 2.46, CS.out.vEgoRaw,
                                           CS.out.steeringAngleDeg, CC.latActive, CarControllerParams, controller.VM)
    reanchored = apply_steer_angle_limits_vm(CC.actuators.steeringAngleDeg, CS.out.steeringAngleDeg, CS.out.vEgoRaw,
                                             CS.out.steeringAngleDeg, CC.latActive, CarControllerParams, controller.VM)

    self.assertAlmostEqual(controller.apply_angle_last, expected)
    self.assertNotAlmostEqual(controller.apply_angle_last, reanchored)

  def test_lkas_angle_inactive_tracks_live_angle(self):
    controller = self._controller()
    controller.apply_angle_last = -15.0
    controller.lat_active_prev = True

    CS = self._state(0.0, 24.5)
    CC = self._control(False, -40.0)
    controller.handle_angle_lateral(CC, CS)

    self.assertEqual(controller.apply_angle_last, CS.out.steeringAngleDeg)
    self.assertFalse(controller.lat_active_prev)


if __name__ == "__main__":
  unittest.main()
