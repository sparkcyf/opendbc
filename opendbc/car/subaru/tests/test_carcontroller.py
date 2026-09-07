import unittest
from types import SimpleNamespace

from opendbc.can import CANParser
from opendbc.car import structs
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

  def test_lkas_angle_rising_edge_synchronizes_inactive(self):
    controller = self._controller()
    controller.apply_angle_last = -0.43
    CS = self._state(30.8, -0.62)
    CC = self._control(True, -0.81)

    first = controller.handle_angle_lateral(CC, CS)
    self.assertFalse(first[1][1] & 0x10)
    self.assertEqual(controller.apply_angle_last, CS.out.steeringAngleDeg)

    second = controller.handle_angle_lateral(CC, CS)
    self.assertTrue(second[1][1] & 0x10)
    self.assertAlmostEqual(controller.apply_angle_last, CC.actuators.steeringAngleDeg)

    controller.handle_angle_lateral(self._control(False, 0), CS)
    CS.out.steeringAngleDeg = 1.0
    self.assertFalse(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
    self.assertEqual(controller.apply_angle_last, 1.0)

  def test_lkas_angle_waits_until_reference_is_within_accel_limit(self):
    controller = self._controller()
    CC = self._control(True, 0)
    CS = self._state(35, 40)
    for _ in range(5):
      self.assertFalse(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
      self.assertEqual(controller.apply_angle_last, 40)
    CS.out.steeringAngleDeg = 0
    self.assertFalse(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
    self.assertTrue(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)

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

  def test_gen2_cancel_once_per_received_counter(self):
    controller = self._controller()
    parser = CANParser("subaru_global_2017_generated", [], 0)
    CS = self._state(0, 0)
    CS.es_distance_msg = dict(parser.vl["ES_Distance"])
    CS.es_dashstatus_msg = dict(parser.vl["ES_DashStatus"])
    CS.es_lkas_state_msg = dict(parser.vl["ES_LKAS_State"])
    CS.es_infotainment_msg = dict(parser.vl["ES_Infotainment"])
    CC = structs.CarControl.new_message()
    # Observe an initial frame before requesting cancel; stale frames must not be injected.
    controller.update(CC.as_reader(), CS, 0)
    CC.cruiseControl.cancel = True
    last_lkas_counter = 0
    for counter in range(1, 34):
      CS.es_distance_msg["COUNTER"] = counter % 16
      for repeat in range(5):
        _, sends = controller.update(CC.as_reader(), CS, 0)
        lkas = [msg for msg in sends if msg[0] == 0x124]
        self.assertEqual(len(lkas), controller.frame % 2)
        if lkas:
          last_lkas_counter = (last_lkas_counter + 1) % 16
          self.assertEqual(lkas[0][1][1] & 0xf, last_lkas_counter)
        cancel = [msg for msg in sends if msg[0] == 0x221]
        self.assertEqual(len(cancel), int(repeat == 0))
        if cancel:
          _, data, bus = cancel[0]
          self.assertEqual(bus, 1)
          self.assertEqual(data[1] & 0xf, (counter + 1) % 16)
          self.assertTrue(data[7] & 1)
          self.assertEqual(int.from_bytes(data[2:4], "little") & 0x1fff, 1818)
    CC.cruiseControl.cancel = False
    CS.es_distance_msg["COUNTER"] = 2
    _, sends = controller.update(CC.as_reader(), CS, 0)
    self.assertFalse(any(msg[0] == 0x221 for msg in sends))
    CC.cruiseControl.cancel = True
    _, sends = controller.update(CC.as_reader(), CS, 0)
    self.assertFalse(any(msg[0] == 0x221 for msg in sends))


if __name__ == "__main__":
  unittest.main()
