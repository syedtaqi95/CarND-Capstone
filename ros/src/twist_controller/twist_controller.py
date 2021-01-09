import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel,
                max_steer_angle):
        # TODO: Implement
        
        # Init steering/yaw controller
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel,
                                            max_steer_angle)
        
        # Init throttle/PID controller
        kp = 0.5
        ki = 0.001
        kd = 0.8
        mn = 0. # Min throttle value
        mx = 0.2 # Max throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        # Init velocity/LPF
        tau = 0.5 # 1/(2pi*tau) = cut-off freq
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # Disabled and reset PID if DBW is disabled
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        # Filter out high freq velocity components for smooth acceleration
        current_vel = self.vel_lpf.filt(current_vel)

        # Calculate steering value using yaw controller
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # Calculate velocity value using PID controller
        vel_err = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_err, sample_time)

        # Calculate brake value
        brake = 0
        
        # Apply brake if almost at stop
        if linear_vel == 0 and current_vel < 0.1:
            throttle = 0.
            brake = 700 # Nm to make the car stationary
        
        # if throttle is small, apply brake
        elif throttle < 0.1 and vel_err < 0:
            throttle = 0.
            decel = max(vel_err, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque per m
        
        return throttle, brake, steering
