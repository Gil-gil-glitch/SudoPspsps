#
## dynamixel_driver.py
#
#  Driver for the Dynamixel pan-tilt mechanism on the cat head.
#

from dynamixel_sdk import PortHandler, PacketHandler
import time

# XL430 Control Table Addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

PROTOCOL_VERSION = 2.0

TICKS_PER_DEGREE = 4095.0 / 360.0


class DynamixelDriver:

    def __init__(
        self,
        device_name="/dev/ttyACM0",
        baudrate=1000000,
        pan_id=1,
        tilt_id=2,
    ):

        self.pan_id = pan_id
        self.tilt_id = tilt_id

        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise RuntimeError(
                f"Failed to open port {device_name}"
            )

        if not self.port_handler.setBaudRate(baudrate):
            raise RuntimeError(
                f"Failed to set baudrate {baudrate}"
            )

        print(f"Connected to {device_name}")

        self.pan_zero = 0
        self.tilt_zero = 0

    # Communication Helpers
    def ping(self, dxl_id):

        model, comm_result, error = (
            self.packet_handler.ping(
                self.port_handler,
                dxl_id
            )
        )

        print(
            f"PING ID={dxl_id} "
            f"MODEL={model} "
            f"COMM={comm_result} "
            f"ERROR={error}"
        )

        return comm_result == 0

    # Torque
    def enable_torque(self, dxl_id):

        comm_result, error = (
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_TORQUE_ENABLE,
                TORQUE_ENABLE,
            )
        )

        print(
            f"Enable Torque ID={dxl_id} "
            f"COMM={comm_result} "
            f"ERROR={error}"
        )

    def disable_torque(self, dxl_id):

        self.packet_handler.write1ByteTxRx(
            self.port_handler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE,
        )

    def enable(self):

        self.enable_torque(self.pan_id)
        self.enable_torque(self.tilt_id)

    def disable(self):

        self.disable_torque(self.pan_id)
        self.disable_torque(self.tilt_id)

    # Raw Position Access
    def read_position(self, dxl_id):

        position, comm_result, error = (
            self.packet_handler.read4ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_PRESENT_POSITION,
            )
        )

        if comm_result != 0:
            print(
                f"Read Error ID={dxl_id} "
                f"COMM={comm_result}"
            )

        return position

    def write_position(self, dxl_id, position):

        comm_result, error = (
            self.packet_handler.write4ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_GOAL_POSITION,
                int(position),
            )
        )

        #print(
        #    f"Write ID={dxl_id} "
        #    f"POS={position} "
        #    f"COMM={comm_result} "
        #    f"ERROR={error}"
        #)

    # Calibration
    def calibrate_zero(self):

        self.pan_zero = self.read_position(
            self.pan_id
        )

        self.tilt_zero = self.read_position(
            self.tilt_id
        )

        print()
        print("=== ZERO CALIBRATION ===")
        print(f"Pan Zero  : {self.pan_zero}")
        print(f"Tilt Zero : {self.tilt_zero}")
        print("========================")
        print()

    # Safe Limits
    PAN_MIN = -60
    PAN_MAX = 60

    TILT_MIN = -15
    TILT_MAX = 20

    # Pan Control
    def set_pan(self, angle):

        angle = max(
            self.PAN_MIN,
            min(self.PAN_MAX, angle)
        )

        position = int(
            self.pan_zero +
            angle * TICKS_PER_DEGREE
        )

        self.write_position(
            self.pan_id,
            position
        )

    def get_pan(self):

        position = self.read_position(
            self.pan_id
        )

        return (
            position - self.pan_zero
        ) / TICKS_PER_DEGREE

    # Tilt Control
    def set_tilt(self, angle):

        angle = max(
            self.TILT_MIN,
            min(self.TILT_MAX, angle)
        )

        position = int(
            self.tilt_zero +
            angle * TICKS_PER_DEGREE
        )

        self.write_position(
            self.tilt_id,
            position
        )

    def get_tilt(self):

        position = self.read_position(
            self.tilt_id
        )

        return (
            position - self.tilt_zero
        ) / TICKS_PER_DEGREE

    # Cleanup
    def close(self):

        self.port_handler.closePort()