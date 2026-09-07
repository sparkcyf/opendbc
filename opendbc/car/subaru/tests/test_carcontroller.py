import math
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
    return SimpleNamespace(out=SimpleNamespace(vEgoRaw=speed, steeringAngleDeg=steering_angle,
                                               cruiseState=SimpleNamespace(enabled=True), steerFaultTemporary=False))

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
    self.assertEqual(controller.angle_filter.x, controller.apply_angle_last)
    self.assertFalse(controller.lat_active_prev)

  def test_angle_filter_small_correction_and_time_response(self):
    controller = self._controller()
    CS = self._state(1.0, 0.0)
    CC = self._control(True, 4.0)  # Below the previous low-speed deadzone.
    self.assertFalse(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
    for _ in range(5):  # Five 50 Hz updates are 100 ms.
      controller.handle_angle_lateral(CC, CS)
    self.assertAlmostEqual(controller.apply_angle_last / 4.0, 1 - (5 / 6)**5)
    for _ in range(45):
      controller.handle_angle_lateral(CC, CS)
    self.assertAlmostEqual(controller.apply_angle_last, 4.0, places=3)

  def test_angle_filter_weakens_with_speed_and_bypasses(self):
    first_steps = []
    for speed in (0.0, 2.0, 6.0, 9.999, 10.0, 10.001):
      controller = self._controller()
      CS = self._state(speed, 0.0)
      CC = self._control(True, 1.0)
      controller.handle_angle_lateral(CC, CS)
      controller.handle_angle_lateral(CC, CS)
      first_steps.append(controller.apply_angle_last)
    self.assertEqual(first_steps[0], first_steps[1])
    self.assertLess(first_steps[1], first_steps[2])
    self.assertLess(first_steps[2], first_steps[3])
    self.assertAlmostEqual(first_steps[3], first_steps[4], delta=0.003)
    self.assertEqual(first_steps[4:], [1.0, 1.0])

  def test_angle_filter_speed_transition_tracks_current_target(self):
    controller = self._controller()
    CS = self._state(2.0, 0.0)
    CC = self._control(True, 1.0)
    controller.handle_angle_lateral(CC, CS)
    controller.handle_angle_lateral(CC, CS)
    CS.out.vEgoRaw = 10.0
    controller.handle_angle_lateral(CC, CS)
    self.assertEqual(controller.apply_angle_last, 1.0)
    CC.actuators.steeringAngleDeg = -0.5
    controller.handle_angle_lateral(CC, CS)
    self.assertEqual(controller.apply_angle_last, -0.5)
    CS.out.vEgoRaw = 2.0
    controller.handle_angle_lateral(CC, CS)
    self.assertEqual(controller.apply_angle_last, -0.5)

  def test_angle_filter_attenuates_repeated_requests(self):
    controller = self._controller()
    CS = self._state(1.0, 0.0)
    controller.handle_angle_lateral(self._control(True, 0.0), CS)
    requested, sent = [], []
    for frame in range(200):
      target = 3 * math.sin(2 * math.pi * 2.5 * frame * 0.02)
      controller.handle_angle_lateral(self._control(True, target), CS)
      if frame >= 100:
        requested.append(target)
        sent.append(controller.apply_angle_last)
    amplitude_ratio = math.sqrt(sum(a*a for a in sent) / sum(a*a for a in requested))
    self.assertGreater(amplitude_ratio, 0.4)
    self.assertLess(amplitude_ratio, 0.6)

  def test_angle_filter_resets_on_every_inactive_frame(self):
    for exit_reason in ("disengage", "fault", "cruise", "accel_limit"):
      with self.subTest(exit_reason=exit_reason):
        controller = self._controller()
        CS = self._state(1.0, 0.0)
        CC = self._control(True, 100.0)
        for _ in range(30):
          controller.handle_angle_lateral(CC, CS)
        if exit_reason == "disengage":
          CC.latActive = False
        elif exit_reason == "fault":
          CS.out.steerFaultTemporary = True
        elif exit_reason == "cruise":
          CS.out.cruiseState.enabled = False
        else:
          CS.out.vEgoRaw = 40.0
        for measured in (-30.0, -40.0):
          CS.out.steeringAngleDeg = measured
          self.assertFalse(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
          self.assertEqual(controller.apply_angle_last, measured)
          self.assertEqual(controller.angle_filter.x, measured)
        CS = self._state(1.0, -40.0)
        CC = self._control(True, -40.0)
        for _ in range(3):
          controller.handle_angle_lateral(CC, CS)
          self.assertEqual(controller.apply_angle_last, -40.0)

  def test_filtered_commands_pass_panda_limits(self):
    from opendbc.safety.tests.libsafety.libsafety_py import make_CANPacket
    from opendbc.safety.tests.test_subaru import TestSubaruGen2AngleStockLongitudinalSafety

    safety_case = TestSubaruGen2AngleStockLongitudinalSafety()
    for speed in (0.0, 2.0, 6.0, 9.99, 10.0, 10.01, 20.0, 40.0):
      with self.subTest(speed=speed):
        safety_case.setUp()
        safety_case._reset_speed_measurement(speed)
        safety_case.safety.set_controls_allowed(True)
        controller = self._controller()
        for frame in range(200):
          safety_case.safety.set_timer(frame * 20000)
          measured = controller.apply_angle_last
          safety_case._rx(safety_case._angle_meas_msg(measured))
          CS = self._state(speed, measured)
          CC = self._control(frame % 50 != 0, 250.0 if frame % 100 < 50 else -250.0)
          addr, data, bus = controller.handle_angle_lateral(CC, CS)
          self.assertTrue(safety_case._tx(make_CANPacket(addr, bus, data)))

  def test_fault_or_cruise_exit_overrides_stale_active_request(self):
    for fault, cruise in ((True, True), (False, False)):
      with self.subTest(fault=fault, cruise=cruise):
        controller = self._controller()
        CS = self._state(30.8, 0)
        CC = self._control(True, -1.2)
        controller.handle_angle_lateral(CC, CS)
        self.assertTrue(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
        CS.out.steerFaultTemporary = fault
        CS.out.cruiseState.enabled = cruise
        for measured in (0.1, -0.2, 0.5):
          CS.out.steeringAngleDeg = measured
          self.assertFalse(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
          self.assertEqual(controller.apply_angle_last, measured)
          self.assertEqual(controller.angle_filter.x, measured)
        CS.out.steerFaultTemporary = False
        CS.out.cruiseState.enabled = True
        self.assertFalse(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)
        self.assertTrue(controller.handle_angle_lateral(CC, CS)[1][1] & 0x10)

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
