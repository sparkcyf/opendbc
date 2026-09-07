import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.lateral import (apply_driver_steer_torque_limits, apply_steer_angle_limits_vm,
                                 common_fault_avoidance, get_max_angle_vm)
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CarControllerParams, SubaruFlags
from opendbc.car.vehicle_model import VehicleModel

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut


def get_safety_cp():
  # Match Panda safety's conservative Ascent vehicle model (the angle Subaru with the most restrictive slip factor).
  from opendbc.car.subaru.interface import CarInterface
  return CarInterface.get_non_essential_params("SUBARU_ASCENT")


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_torque_last = 0
    self.apply_angle_last = 0.0
    self.lat_active_prev = False
    self.es_distance_counter_last = -1

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0

    self.p = CarControllerParams(CP)
    self.angle_filter = FirstOrderFilter(0.0, 0.1, DT_CTRL * self.p.STEER_STEP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])
    self.VM = VehicleModel(get_safety_cp()) if CP.flags & SubaruFlags.LKAS_ANGLE else None

  def handle_angle_lateral(self, CC, CS):
    # Synchronize with safety using an inactive command before engaging. Re-anchoring only the
    # controller on the first active frame can violate safety's rate limit from the previous command.
    # Act on the latest state even if carControl has not yet reacted to a rejected
    # TX or cruise exit. Continue sending inactive measured-angle messages at 50 Hz.
    lat_requested = CC.latActive and CS.out.cruiseState.enabled and not CS.out.steerFaultTemporary
    lat_active = lat_requested and self.lat_active_prev
    assert self.VM is not None
    max_angle = get_max_angle_vm(max(CS.out.vEgoRaw, 1), self.VM, CarControllerParams)
    # An inactive reference can be outside the speed-dependent limit. Keep tracking the wheel until
    # it is possible to engage without violating either the acceleration or the jerk limit.
    lat_active = lat_active and abs(self.apply_angle_last) <= max_angle

    apply_angle = CC.actuators.steeringAngleDeg
    # Smooth low-speed requests without holding small corrections. Fade out by 10 m/s
    # to avoid adding high-speed lag; final vehicle-model and rate limits still apply.
    if lat_active:
      self.angle_filter.update_alpha(float(np.interp(CS.out.vEgoRaw, [2.0, 10.0], [0.1, 0.0])))
      apply_angle = self.angle_filter.update(apply_angle)

    self.apply_angle_last = apply_steer_angle_limits_vm(apply_angle, self.apply_angle_last, CS.out.vEgoRaw,
                                                        CS.out.steeringAngleDeg, lat_active, CarControllerParams, self.VM)
    if not lat_active:
      # Seed from the inactive command, including the synchronization frame before engagement.
      self.angle_filter.x = self.apply_angle_last
    self.lat_active_prev = lat_requested
    return subarucan.create_steering_control_angle(self.packer, self.apply_angle_last, lat_active)

  def handle_torque_lateral(self, CC, CS):
    apply_torque = int(round(CC.actuators.torque * self.p.STEER_MAX))

    # Limits due to driver torque
    apply_torque = apply_driver_steer_torque_limits(apply_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

    if not CC.latActive:
      apply_torque = 0

    if self.CP.flags & SubaruFlags.PREGLOBAL:
      msg = subarucan.create_preglobal_steering_control(self.packer, self.frame // self.p.STEER_STEP, apply_torque, CC.latActive)
    else:
      apply_steer_req = CC.latActive

      if self.CP.flags & SubaruFlags.STEER_RATE_LIMITED:
        # Steering rate fault prevention
        self.steer_rate_counter, apply_steer_req = \
          common_fault_avoidance(abs(CS.out.steeringRateDeg) > MAX_STEER_RATE, apply_steer_req,
                                 self.steer_rate_counter, MAX_STEER_RATE_FRAMES)

      msg = subarucan.create_steering_control(self.packer, apply_torque, apply_steer_req)

    self.apply_torque_last = apply_torque
    return msg

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
    if (self.frame % self.p.STEER_STEP) == 0:
      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        can_sends.append(self.handle_angle_lateral(CC, CS))
      else:
        can_sends.append(self.handle_torque_lateral(CC, CS))

    # *** longitudinal ***

    if CC.longActive:
      apply_throttle = int(round(np.interp(actuators.accel, CarControllerParams.THROTTLE_LOOKUP_BP, CarControllerParams.THROTTLE_LOOKUP_V)))
      apply_rpm = int(round(np.interp(actuators.accel, CarControllerParams.RPM_LOOKUP_BP, CarControllerParams.RPM_LOOKUP_V)))
      apply_brake = int(round(np.interp(actuators.accel, CarControllerParams.BRAKE_LOOKUP_BP, CarControllerParams.BRAKE_LOOKUP_V)))

      # limit min and max values
      cruise_throttle = np.clip(apply_throttle, CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
      cruise_rpm = np.clip(apply_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
      cruise_brake = np.clip(apply_brake, CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
    else:
      cruise_throttle = CarControllerParams.THROTTLE_INACTIVE
      cruise_rpm = CarControllerParams.RPM_MIN
      cruise_brake = CarControllerParams.BRAKE_MIN

    # *** alerts and pcm cancel ***
    if self.CP.flags & SubaruFlags.PREGLOBAL:
      if self.frame % 5 == 0:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))

    else:
      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, CC.enabled,
                                                        self.CP.openpilotLongitudinalControl, CC.longActive, hud_control.leadVisible))

        can_sends.append(subarucan.create_es_lkas_state(self.packer, self.frame // 10, CS.es_lkas_state_msg, CC.enabled, hud_control.visualAlert,
                                                        hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                        hud_control.leftLaneDepart, hud_control.rightLaneDepart))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_es_infotainment(self.packer, self.frame // 10, CS.es_infotainment_msg, hud_control.visualAlert))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, self.frame // 5, CS.es_status_msg,
                                                      self.CP.openpilotLongitudinalControl, CC.longActive, cruise_rpm))

          can_sends.append(subarucan.create_es_brake(self.packer, self.frame // 5, CS.es_brake_msg,
                                                     self.CP.openpilotLongitudinalControl, CC.longActive, cruise_brake))

          can_sends.append(subarucan.create_es_distance(self.packer, self.frame // 5, CS.es_distance_msg, 0, pcm_cancel_cmd,
                                                        self.CP.openpilotLongitudinalControl, cruise_brake > 0, cruise_throttle))
      else:
        if not (self.CP.flags & SubaruFlags.HYBRID):
          counter = CS.es_distance_msg["COUNTER"]
          # Gen2 injects cancel on the alternate bus alongside EyeSight. Do not repeatedly inject
          # the same counter at the 100 Hz control rate while waiting for a new stock message.
          new_distance = counter != self.es_distance_counter_last
          if pcm_cancel_cmd and (not (self.CP.flags & SubaruFlags.GLOBAL_GEN2) or new_distance):
            bus = CanBus.alt if self.CP.flags & SubaruFlags.GLOBAL_GEN2 else CanBus.main
            can_sends.append(subarucan.create_es_distance(self.packer, counter + 1, CS.es_distance_msg, bus, pcm_cancel_cmd))
          self.es_distance_counter_last = counter

      if self.CP.flags & SubaruFlags.DISABLE_EYESIGHT:
        # Tester present (keeps eyesight disabled)
        if self.frame % 100 == 0:
          can_sends.append(make_tester_present_msg(GLOBAL_ES_ADDR, CanBus.camera, suppress_response=True))

        # Create all of the other eyesight messages to keep the rest of the car happy when eyesight is disabled
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_highbeamassist(self.packer))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_static_1(self.packer))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_es_static_2(self.packer))

    new_actuators = actuators.as_builder()
    if self.CP.flags & SubaruFlags.LKAS_ANGLE:
      new_actuators.steeringAngleDeg = self.apply_angle_last
    else:
      new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
      new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends
