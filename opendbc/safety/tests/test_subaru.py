#!/usr/bin/env python3
import enum
import unittest
from functools import partial
from itertools import product

import numpy as np

from opendbc.car import CanData, structs
from opendbc.car.lateral import get_max_angle_vm
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, CarControllerParams
from opendbc.car.subaru.values import SubaruSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.car.vehicle_model import VehicleModel
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import away_round, round_speed


class SubaruMsg(enum.IntEnum):
  Brake_Status      = 0x13c
  CruiseControl     = 0x240
  Throttle          = 0x40
  Steering_2        = 0x11a
  Steering_Torque   = 0x119
  Wheel_Speeds      = 0x13a
  ES_LKAS           = 0x122
  ES_LKAS_ANGLE     = 0x124
  ES_Distance       = 0x221
  ES_Status         = 0x222
  ES_DashStatus     = 0x321
  ES_LKAS_State     = 0x322
  ES_Infotainment   = 0x323


SUBARU_MAIN_BUS = 0
SUBARU_ALT_BUS  = 1
SUBARU_CAM_BUS  = 2


def lkas_tx_msgs(alt_bus, lkas_msg=SubaruMsg.ES_LKAS):
  return [[lkas_msg,                    SUBARU_MAIN_BUS],
          [SubaruMsg.ES_Distance,       alt_bus],
          [SubaruMsg.ES_DashStatus,     SUBARU_MAIN_BUS],
          [SubaruMsg.ES_LKAS_State,     SUBARU_MAIN_BUS],
          [SubaruMsg.ES_Infotainment,   SUBARU_MAIN_BUS]]


def fwd_blacklisted_addr(lkas_msg=SubaruMsg.ES_LKAS):
  return {SUBARU_CAM_BUS: [lkas_msg, SubaruMsg.ES_DashStatus, SubaruMsg.ES_LKAS_State, SubaruMsg.ES_Infotainment]}


class TestSubaruSafetyBase(common.CarSafetyTest):
  DBC = "subaru_global_2017_generated"
  SAFETY_MODEL = CarParams.SafetyModel.subaru

  RELAY_MALFUNCTION_ADDRS = {SUBARU_MAIN_BUS: (SubaruMsg.ES_LKAS, SubaruMsg.ES_DashStatus, SubaruMsg.ES_LKAS_State,
                                               SubaruMsg.ES_Infotainment)}
  FWD_BLACKLISTED_ADDRS = fwd_blacklisted_addr()

  MAX_RT_DELTA = 940

  DRIVER_TORQUE_ALLOWANCE = 60
  DRIVER_TORQUE_FACTOR = 50

  ALT_MAIN_BUS = SUBARU_MAIN_BUS
  ALT_CAM_BUS = SUBARU_CAM_BUS

  DEG_TO_CAN = 100

  INACTIVE_GAS = 1818

  def _set_prev_torque(self, t):
    self.safety.set_desired_torque_last(t)
    self.safety.set_rt_torque_last(t)

  def _torque_driver_msg(self, torque):
    values = {"Steer_Torque_Sensor": torque}
    return self.packer.make_can_msg_safety("Steering_Torque", 0, values)

  def _speed_msg(self, speed, wheels=("FR", "FL", "RR", "RL")):
    values = {s: speed if s in wheels else 0 for s in ["FR", "FL", "RR", "RL"]}
    return self.packer.make_can_msg_safety("Wheel_Speeds", self.ALT_MAIN_BUS, values)

  def _user_brake_msg(self, brake):
    values = {"Brake": brake}
    return self.packer.make_can_msg_safety("Brake_Status", self.ALT_MAIN_BUS, values)

  def _user_gas_msg(self, gas):
    values = {"Throttle_Pedal": gas}
    return self.packer.make_can_msg_safety("Throttle", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"Cruise_Activated": enable}
    return self.packer.make_can_msg_safety("CruiseControl", self.ALT_MAIN_BUS, values)

  def test_vehicle_moving_any_wheel(self):
    # any single wheel reporting a non-zero speed means the vehicle is moving
    for wheel in ["FR", "FL", "RR", "RL"]:
      self._rx(self._speed_msg(0))
      self.assertFalse(self.safety.get_vehicle_moving())
      self._rx(self._speed_msg(1, wheels=(wheel,)))
      self.assertTrue(self.safety.get_vehicle_moving())


class TestSubaruStockLongitudinalSafetyBase(TestSubaruSafetyBase):
  def _cancel_msg(self, cancel, cruise_throttle=0):
    values = {"Cruise_Cancel": cancel, "Cruise_Throttle": cruise_throttle}
    return self.packer.make_can_msg_safety("ES_Distance", self.ALT_MAIN_BUS, values)

  def test_cancel_message(self):
    # test that we can only send the cancel message (ES_Distance) with inactive throttle (1818) and Cruise_Cancel=1
    for cancel in [True, False]:
      self._generic_limit_safety_check(partial(self._cancel_msg, cancel), self.INACTIVE_GAS, self.INACTIVE_GAS, 0, 2**12, 1, self.INACTIVE_GAS, cancel)


class TestSubaruTorqueSafetyBase(TestSubaruSafetyBase, common.DriverTorqueSteeringSafetyTest, common.SteerRequestCutSafetyTest):
  MAX_RATE_UP = 50
  MAX_RATE_DOWN = 70
  MAX_TORQUE_LOOKUP = [0], [2047]

  # Safety around steering req bit
  MIN_VALID_STEERING_FRAMES = 7
  MAX_INVALID_STEERING_FRAMES = 1
  STEER_STEP = 2

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAS_Output": torque, "LKAS_Request": steer_req}
    return self.packer.make_can_msg_safety("ES_LKAS", SUBARU_MAIN_BUS, values)


class TestSubaruGen1TorqueStockLongitudinalSafety(TestSubaruStockLongitudinalSafetyBase, TestSubaruTorqueSafetyBase):
  TX_MSGS = lkas_tx_msgs(SUBARU_MAIN_BUS)


class TestSubaruGen2TorqueSafetyBase(TestSubaruTorqueSafetyBase):
  ALT_MAIN_BUS = SUBARU_ALT_BUS
  ALT_CAM_BUS = SUBARU_ALT_BUS

  MAX_RATE_UP = 40
  MAX_RATE_DOWN = 40
  MAX_TORQUE_LOOKUP = [0], [1000]


class TestSubaruGen2TorqueStockLongitudinalSafety(TestSubaruStockLongitudinalSafetyBase, TestSubaruGen2TorqueSafetyBase):
  SAFETY_PARAM = SubaruSafetyFlags.GEN2

  TX_MSGS = lkas_tx_msgs(SUBARU_ALT_BUS)


class TestSubaruAngleSafetyBase(TestSubaruSafetyBase, common.AngleSteeringSafetyTest):
  STEER_ANGLE_MAX = 190
  DEG_TO_CAN = 100

  # Vehicle-model limits are tested explicitly below instead of using breakpoint-based generic tests.
  LATERAL_FREQUENCY = 50

  def setUp(self):
    super().setUp()
    self.cnt_angle_cmd = 0
    from opendbc.car.subaru.carcontroller import get_safety_cp
    self.VM = VehicleModel(get_safety_cp())

  def _speed_msg(self, speed, wheels=("FR", "FL", "RR", "RL")):
    # Angle safety tests use m/s; the Subaru wheel-speed DBC signals use km/h.
    speed_kph = speed * 3.6
    values = {s: speed_kph if s in wheels else 0 for s in ["FR", "FL", "RR", "RL"]}
    return self.packer.make_can_msg_safety("Wheel_Speeds", self.ALT_MAIN_BUS, values)

  def _angle_cmd_msg(self, angle, enabled, increment_timer=True):
    values = {"LKAS_Output": angle, "LKAS_Request": enabled, "SET_3": 3}
    if increment_timer:
      self.safety.set_timer(self.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.cnt_angle_cmd += 1
    return self.packer.make_can_msg_safety("ES_LKAS_ANGLE", SUBARU_MAIN_BUS, values)

  def _angle_meas_msg(self, angle):
    values = {"Steering_Angle": angle}
    return self.packer.make_can_msg_safety("Steering_2", SUBARU_MAIN_BUS, values)

  def _pcm_status_msg(self, enable):
    values = {"Cruise_Activated": enable}
    return self.packer.make_can_msg_safety("ES_Status", self.ALT_MAIN_BUS, values)

  def test_angle_cmd_when_enabled(self):
    # The generic test assumes breakpoint-based limits. Vehicle-model acceleration and jerk are tested below.
    pass

  def test_active_angle_hard_limits(self):
    self._reset_speed_measurement(1)
    for sign in (-1, 1):
      for angle in (190, 190.01):
        self.safety.set_controls_allowed(True)
        self.safety.set_desired_angle_last(round(angle * sign * self.DEG_TO_CAN))
        self.assertEqual(angle <= 190, self._tx(self._angle_cmd_msg(angle * sign, True)))
      for delta in (5, 5.01, 5.02, 10):
        self.safety.set_controls_allowed(True)
        self.safety.set_desired_angle_last(0)
        self.assertEqual(delta <= 5.01, self._tx(self._angle_cmd_msg(delta * sign, True)))
        if delta > 5.01:
          # History resets to the measured angle: only passes the rate limit from 0, not from the rejected command
          self.assertTrue(self._tx(self._angle_cmd_msg(-4 * sign, True)))

  def test_controller_engagement_passes_safety(self):
    from types import SimpleNamespace
    from opendbc.car.subaru.carcontroller import CarController
    from opendbc.car.subaru.interface import CarInterface
    from opendbc.car.subaru.values import CAR

    for speed in (5, 15, 30.8, 40):
      for sign in (-1, 1):
        self._reset_speed_measurement(speed)
        self.safety.set_controls_allowed(True)
        self.safety.set_desired_angle_last(0)
        controller = CarController({}, CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023))
        CC = SimpleNamespace(latActive=False, actuators=SimpleNamespace(steeringAngleDeg=0))
        CS = SimpleNamespace(out=SimpleNamespace(vEgoRaw=speed, steeringAngleDeg=0,
                                                 cruiseState=SimpleNamespace(enabled=True), steerFaultTemporary=False))
        for frame in range(30):
          # ACC engages between two angle samples. The wheel also moves during the handoff.
          CS.out.steeringAngleDeg = sign * (0.43 if frame == 0 else 0.62 + min(frame - 1, 3) * 0.15)
          self._rx(self._angle_meas_msg(CS.out.steeringAngleDeg))
          CC.latActive = 1 <= frame < 20 or frame >= 23
          CC.actuators.steeringAngleDeg = sign * 3
          addr, data, bus = controller.handle_angle_lateral(CC, CS)
          self.safety.set_timer(frame * 20000)
          self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, data)), (speed, sign, frame))

  def _make_angle_controller(self, speed, angle, target, cruise_enabled=True):
    from types import SimpleNamespace
    from opendbc.car.subaru.carcontroller import CarController

    controller = CarController({}, CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023))
    CC = SimpleNamespace(latActive=cruise_enabled, actuators=SimpleNamespace(steeringAngleDeg=target))
    CS = SimpleNamespace(out=SimpleNamespace(vEgoRaw=speed, steeringAngleDeg=angle, steerFaultTemporary=False,
                                             cruiseState=SimpleNamespace(enabled=cruise_enabled)))
    return controller, CC, CS

  def test_controller_engage_while_steering(self):
    # Panda receives each Steering_2 sample before CarState does. When the driver is turning while
    # engaging, panda's latest angle is one sample ahead of the angle the controller limits from.
    engage_frame, total_frames = 52, 150
    for speed in (5., 15., 26., 35.):
      for rate in (-100., -16., 16., 100.):
        with self.subTest(speed=speed, rate=rate):
          self.setUp()
          self._reset_speed_measurement(speed)
          angle = angle_prev = -rate * 0.05
          controller, CC, CS = self._make_angle_controller(speed, angle, 0., cruise_enabled=False)
          for frame in range(0, total_frames, 2):
            angle_prev = angle
            if engage_frame - 10 <= frame < engage_frame:
              angle += rate * 0.02
            self._rx(self._angle_meas_msg(angle))
            CS.out.steeringAngleDeg = CC.actuators.steeringAngleDeg = angle_prev
            CS.out.cruiseState.enabled = CC.latActive = frame >= engage_frame
            self.safety.set_controls_allowed(CC.latActive)
            self.safety.set_timer(frame * 10000)
            addr, data, bus = controller.handle_angle_lateral(CC, CS)
            self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, data)), frame)

  def test_controller_keeps_request_at_accel_limit(self):
    # Wheel-speed noise moves the limit past the last command while a curve demands more than the limit
    for speed in (10., 15.7, 20., 30.):
      with self.subTest(speed=speed):
        self.setUp()
        self._reset_speed_measurement(speed)
        self.safety.set_controls_allowed(True)
        max_angle = get_max_angle_vm(speed, self.VM, CarControllerParams)
        angle = 0.8 * max_angle
        controller, CC, CS = self._make_angle_controller(speed, angle, 1.15 * max_angle)
        self._rx(self._angle_meas_msg(angle))
        for frame in range(400):
          CS.out.vEgoRaw = speed + (0.03 if frame % 5 == 0 else -0.03 if frame % 7 == 0 else 0.)
          CS.out.steeringAngleDeg = angle
          self._rx(self._angle_meas_msg(angle))
          self.safety.set_timer(frame * 20000)
          addr, data, bus = controller.handle_angle_lateral(CC, CS)
          self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, data)), frame)
          # The first frame after creating the controller is the synchronization frame
          self.assertEqual(bool(data[1] & 0x10), frame > 0, frame)
          angle += 0.2 * (controller.apply_angle_last - angle)

  def test_controller_accelerating_at_accel_limit_passes_safety(self):
    # The limit can fall faster than the jerk limit allows steering back: drop the request before safety rejects
    for speed in (3., 10., 25.):
      for accel in (3., 6.):
        for sign in (-1, 1):
          with self.subTest(speed=speed, accel=accel, sign=sign):
            self.setUp()
            self._reset_speed_measurement(speed)
            self.safety.set_controls_allowed(True)
            angle = 0.
            target = sign * min(get_max_angle_vm(speed, self.VM, CarControllerParams), 150.)
            controller, CC, CS = self._make_angle_controller(speed, angle, target)
            for frame in range(150):
              CS.out.vEgoRaw += accel * 0.02
              CS.out.steeringAngleDeg = angle
              self._reset_speed_measurement(CS.out.vEgoRaw)
              self._rx(self._angle_meas_msg(angle))
              self.safety.set_timer(frame * 20000)
              addr, data, bus = controller.handle_angle_lateral(CC, CS)
              self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, data)), frame)
              if data[1] & 0x10:
                angle = controller.apply_angle_last

  def _find_max_allowed_angle_can(self, sign):
    lo, hi = 0, self.STEER_ANGLE_MAX * self.DEG_TO_CAN + 10
    while lo < hi:
      mid = (lo + hi + 1) // 2
      self.safety.set_desired_angle_last(mid * sign)
      if self._tx(self._angle_cmd_msg(mid / self.DEG_TO_CAN * sign, True)):
        lo = mid
      else:
        hi = mid - 1
    return lo

  def test_lateral_accel_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = max(speed, 1)
      # Match Wheel_Speeds quantization (0.057 km/h per CAN unit).
      speed = round_speed(away_round(speed * 3.6 / 0.057) * 0.057 / 3.6)
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)
        self._tx(self._angle_cmd_msg(0, True))

        max_angle_can = self._find_max_allowed_angle_can(sign)

        self.safety.set_desired_angle_last(max_angle_can * sign)
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_can / self.DEG_TO_CAN * sign, True)))

        above_limit_can = max_angle_can + 1
        self.safety.set_desired_angle_last(above_limit_can * sign)
        self._tx(self._angle_cmd_msg(above_limit_can / self.DEG_TO_CAN * sign, True))

        # One CAN unit beyond either the model or EPS limit must be rejected, even at low speed.
        self.assertFalse(self._tx(self._angle_cmd_msg(above_limit_can / self.DEG_TO_CAN * sign, True)))

  def _find_max_allowed_delta_can(self, sign):
    lo, hi = 0, self.STEER_ANGLE_MAX * self.DEG_TO_CAN
    while lo < hi:
      mid = (lo + hi + 1) // 2
      self.safety.set_desired_angle_last(0)
      if self._tx(self._angle_cmd_msg(mid / self.DEG_TO_CAN * sign, True)):
        lo = mid
      else:
        hi = mid - 1
    return lo

  def test_lateral_jerk_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = max(speed, 1)
      speed = round_speed(away_round(speed * 3.6 / 0.057) * 0.057 / 3.6)
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)
        self._tx(self._angle_cmd_msg(0, True))

        max_delta_can = self._find_max_allowed_delta_can(sign)

        self.safety.set_desired_angle_last(0)
        self.assertTrue(self._tx(self._angle_cmd_msg(max_delta_can / self.DEG_TO_CAN * sign, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_delta_can / self.DEG_TO_CAN * sign, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

        above_delta_can = max_delta_can + 1
        self.assertFalse(self._tx(self._angle_cmd_msg(above_delta_can / self.DEG_TO_CAN * sign, True)))

        self.safety.set_desired_angle_last(round(above_delta_can * sign))
        self.assertTrue(self._tx(self._angle_cmd_msg(above_delta_can / self.DEG_TO_CAN * sign, True)))
        self.assertFalse(self._tx(self._angle_cmd_msg(0, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))


class TestSubaruGen1AngleStockLongitudinalSafety(TestSubaruStockLongitudinalSafetyBase, TestSubaruAngleSafetyBase):
  SAFETY_PARAM = SubaruSafetyFlags.LKAS_ANGLE
  TX_MSGS = lkas_tx_msgs(SUBARU_MAIN_BUS, SubaruMsg.ES_LKAS_ANGLE)
  RELAY_MALFUNCTION_ADDRS = {SUBARU_MAIN_BUS: (SubaruMsg.ES_LKAS_ANGLE, SubaruMsg.ES_DashStatus, SubaruMsg.ES_LKAS_State,
                                               SubaruMsg.ES_Infotainment)}
  FWD_BLACKLISTED_ADDRS = fwd_blacklisted_addr(SubaruMsg.ES_LKAS_ANGLE)


class TestSubaruGen2AngleStockLongitudinalSafety(TestSubaruStockLongitudinalSafetyBase, TestSubaruAngleSafetyBase):
  ALT_MAIN_BUS = SUBARU_ALT_BUS
  SAFETY_PARAM = SubaruSafetyFlags.GEN2 | SubaruSafetyFlags.LKAS_ANGLE
  TX_MSGS = lkas_tx_msgs(SUBARU_ALT_BUS, SubaruMsg.ES_LKAS_ANGLE)
  RELAY_MALFUNCTION_ADDRS = {SUBARU_MAIN_BUS: (SubaruMsg.ES_LKAS_ANGLE, SubaruMsg.ES_DashStatus, SubaruMsg.ES_LKAS_State,
                                               SubaruMsg.ES_Infotainment)}
  FWD_BLACKLISTED_ADDRS = fwd_blacklisted_addr(SubaruMsg.ES_LKAS_ANGLE)

  def test_controller_recovers_rejected_tx_without_active_retry(self):
    for speed, sign, feedback_delay, fault_frame in product((0, 5, 30.8, 40), (-1, 1), (1, 2, 4), (0, 20)):
      with self.subTest(speed=speed, sign=sign, feedback_delay=feedback_delay, fault_frame=fault_frame):
        self.setUp()
        CI = CarInterface(CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023))
        CI.update([])
        CC = structs.CarControl(latActive=True)
        CC.actuators.steeringAngleDeg = sign * 1.2
        pending = []
        last_counter = None
        for frame in range(80):
          self.safety.set_timer(frame * 10000)
          enabled = not 60 <= frame < 66
          measured = sign * (0.1 if frame < 40 else 0.2)
          messages = [
            self.packer.make_can_msg("ES_Status", 1, {"Cruise_Activated": enabled}),
            self.packer.make_can_msg("Steering_2", 0, {"Steering_Angle": measured}),
            self.packer.make_can_msg("Wheel_Speeds", 1, {wheel: speed * 3.6 for wheel in ("FL", "FR", "RL", "RR")}),
          ]
          for addr, data, bus in messages:
            self.assertTrue(self._rx(libsafety_py.make_CANPacket(addr, bus, data)))
          feedback = [msg for due, msg in pending if due == frame]
          CI.update([(frame * 10000000, [*messages, *feedback])])
          # Simulate controlsd reacting late and even trying to re-enable.
          CC.latActive = not 44 <= frame < 48
          if frame == fault_frame == 0:
            CI.CS.out.steeringAngleDeg = sign * 0.62  # Stale first inactive reference, outside Panda's window.
          _, sends = CI.apply(CC.as_reader(), frame * 10000000)
          angle_sends = [msg for msg in sends if msg[0] == 0x124]
          self.assertEqual(len(angle_sends), int(frame % 2 == 0))
          for addr, data, bus in angle_sends:
            counter = data[1] & 0xf
            if last_counter is not None:
              self.assertEqual(counter, (last_counter + 1) % 16)
            last_counter = counter
            active = bool(data[1] & 0x10)
            if fault_frame + feedback_delay <= frame < 66:
              self.assertFalse(active)
              self.assertAlmostEqual(CI.CC.apply_angle_last, measured, places=2)
            elif frame >= 68:
              self.assertTrue(active)
            if frame == 66:
              self.assertFalse(active)  # Fresh engagement must synchronize again.
            if frame == fault_frame == 20:
              # One out-of-bounds TX resets Panda's desired-angle history.
              data = self.packer.make_can_msg("ES_LKAS_ANGLE", 0,
                {"LKAS_Request": 1, "LKAS_Output": sign * 200})[1]
            accepted = self._tx(libsafety_py.make_CANPacket(addr, bus, data))
            if frame == fault_frame:
              self.assertFalse(accepted)
            elif frame >= fault_frame + feedback_delay or frame < fault_frame:
              self.assertTrue(accepted)
            if not accepted:
              pending.append((frame + feedback_delay, CanData(addr, data, 0xC0)))


if __name__ == "__main__":
  unittest.main()
