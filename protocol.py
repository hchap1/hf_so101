"""
Optimized SO-101 robotic arm control with high-performance teleoperation.

This module provides Leader and Follower arm classes with:
- High-frequency position updates (100+ Hz capable)
- Robust error handling and automatic recovery
- Minimal latency through batched operations
- Thread-safe position caching
- Smooth motion through position filtering

Usage:
    from so101_optimized import LeaderArm, FollowerArm, teleoperate

    leader = LeaderArm('/dev/ttyUSB0')
    follower = FollowerArm('/dev/ttyUSB1')

    # Start teleoperation (Ctrl+C to stop)
    teleoperate(leader, follower, update_rate=100)

    # Clean shutdown
    leader.close()
    follower.close()
"""

from typing import Optional
import serial
import threading
import time
import logging
from feetech_servo_protocol import (
    FeetechServobus,
    JOINT_NAMES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

SERVO_IDS = list(range(1, 7))
POSITION_MIN = 0
POSITION_MAX = 4095

# Elbow offset compensation - leader elbow is ~1 rotation offset from follower
# This is likely due to different assembly or calibration
ELBOW_OFFSET = 1.0  # in normalized units (0-1 range)


# ============================================================================
# Position normalization functions
# ============================================================================

def normalize_position(ticks: int, servo_id: int, is_leader: bool = True) -> float:
    """Convert raw encoder ticks to normalized 0..1 position.
    
    The STS3215 servos have 4096 positions per rotation (12-bit resolution).
    Gear ratios don't affect the normalization - they just mean the servo
    can rotate multiple times. We normalize based on the raw encoder range.

    Args:
        ticks: Raw encoder value
        servo_id: Servo ID (1-6)
        is_leader: True for leader arm, False for follower

    Returns:
        Normalized position in range [0, ~1] (can exceed 1 in multi-turn mode)
    """
    # Normalize to 0-1 based on 4096 ticks per rotation
    return ticks / 4096.0


def denormalize_position(normalized: float, servo_id: int, is_follower: bool = True) -> int:
    """Convert normalized 0..1 position to raw encoder ticks.

    Args:
        normalized: Normalized position [0, ~1]
        servo_id: Servo ID (1-6)
        is_follower: True for follower arm, False for leader (not used currently)

    Returns:
        Raw encoder ticks, clamped to valid range
    """
    # Convert from 0-1 back to 0-4095 range
    ticks = int(normalized * 4096.0)
    return max(POSITION_MIN, min(POSITION_MAX, ticks))


# ============================================================================
# Position filter for smooth motion
# ============================================================================

class ExponentialMovingAverage:
    """Simple EMA filter for position smoothing."""

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother
        """
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: float) -> float:
        """Update filter with new value and return filtered output."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        """Reset filter state."""
        self.value = None


# ============================================================================
# Servo abstraction with caching
# ============================================================================

class Servo:
    """Represents a single servo with cached position and thread-safe access."""

    def __init__(self, servo_id: int, name: str, use_filter: bool = False):
        """
        Args:
            servo_id: Servo ID (1-6)
            name: Joint name (yaw, shoulder, etc.)
            use_filter: Enable position filtering for smoother motion
        """
        self.id = servo_id
        self.name = name
        self._position = 0.0  # Normalized position
        self._raw_position = 0  # Raw encoder ticks
        self._lock = threading.Lock()
        self._filter = ExponentialMovingAverage(alpha=0.4) if use_filter else None
        self._last_update_time = 0.0
        self._update_count = 0

    def update_position(self, raw_ticks: int, is_leader: bool = True) -> None:
        """Update cached position from raw encoder reading."""
        normalized = normalize_position(raw_ticks, self.id, is_leader)
        
        if self._filter:
            normalized = self._filter.update(normalized)
        
        with self._lock:
            self._raw_position = raw_ticks
            self._position = normalized
            self._last_update_time = time.perf_counter()
            self._update_count += 1

    def get_position(self) -> float:
        """Get current normalized position (thread-safe)."""
        with self._lock:
            return self._position

    def get_raw_position(self) -> int:
        """Get current raw encoder position (thread-safe)."""
        with self._lock:
            return self._raw_position

    def get_update_rate(self) -> float:
        """Get approximate update rate (Hz)."""
        with self._lock:
            if self._update_count < 10:
                return 0.0
            elapsed = time.perf_counter() - self._last_update_time
            if elapsed > 0:
                return 1.0 / elapsed
            return 0.0


# ============================================================================
# Leader Arm - Read Only
# ============================================================================

class LeaderArm:
    """Leader arm interface with continuous background position reading.

    The leader arm is read-only and continuously updates positions
    in a background thread at high frequency.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        update_rate: float = 100.0,
        reconnect_on_error: bool = True
    ):
        """
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' or 'COM5')
            baudrate: Serial baud rate (default 1000000)
            update_rate: Target update frequency in Hz (default 100)
            reconnect_on_error: Attempt to reconnect on serial errors
        """
        self.port = port
        self.baudrate = baudrate
        self.update_interval = 1.0 / update_rate
        self.reconnect_on_error = reconnect_on_error

        # Initialize serial connection
        self._connect()

        # Create servo objects
        self.servos = {sid: Servo(sid, JOINT_NAMES.get(sid, f"id{sid}")) for sid in SERVO_IDS}

        # Thread control
        self._stop_event = threading.Event()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True, name="LeaderReadThread")

        # Statistics
        self._read_count = 0
        self._error_count = 0
        self._last_successful_read = time.perf_counter()

        # Initial position read
        self._read_all_positions()

        # Start background reading
        self._read_thread.start()
        logger.info(f"Leader arm initialized on {port} at {update_rate} Hz")

    def _connect(self) -> None:
        """Establish serial connection."""
        try:
            self.serial = serial.Serial(
                self.port,
                baudrate=self.baudrate,
                timeout=0.005,
                write_timeout=0.01
            )
            time.sleep(0.05)  # Allow serial port to stabilize
            self.serial.reset_input_buffer()
            self.bus = FeetechServobus(self.serial, max_retries=1)
            logger.info(f"Connected to leader arm on {self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to leader arm: {e}")
            raise

    def _reconnect(self) -> bool:
        """Attempt to reconnect after error."""
        try:
            logger.warning("Attempting to reconnect leader arm...")
            self.serial.close()
            time.sleep(0.5)
            self._connect()
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    def _read_all_positions(self) -> bool:
        """Read positions from all servos using batched operation."""
        try:
            positions = self.bus.read_positions_batch(SERVO_IDS)
            
            success_count = 0
            for servo_id, raw_pos in positions.items():
                if raw_pos is not None:
                    self.servos[servo_id].update_position(raw_pos, is_leader=True)
                    success_count += 1
            
            if success_count > 0:
                self._last_successful_read = time.perf_counter()
                return True
            return False

        except Exception as e:
            logger.error(f"Error reading positions: {e}")
            self._error_count += 1
            return False

    def _read_loop(self) -> None:
        """Background thread continuously reading servo positions."""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            success = self._read_all_positions()

            if success:
                self._read_count += 1
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                logger.warning(f"Read failed (consecutive: {consecutive_errors})")

                if consecutive_errors >= max_consecutive_errors:
                    if self.reconnect_on_error:
                        if self._reconnect():
                            consecutive_errors = 0
                        else:
                            time.sleep(1.0)
                    else:
                        logger.error("Max consecutive errors reached, stopping read loop")
                        break

            # Maintain update rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self.update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Leader read loop stopped")

    def get_positions(self) -> dict[str, float]:
        """Get current normalized positions of all joints.

        Returns:
            dict mapping joint name to normalized position [0, 1]
        """
        positions = {servo.name: servo.get_position() for servo in self.servos.values()}
        
        # Apply elbow offset compensation (leader is ~1 rotation offset from follower)
        if 'elbow' in positions:
            positions['elbow'] -= ELBOW_OFFSET
        
        return positions

    def get_raw_positions(self) -> dict[int, int]:
        """Get raw encoder positions.

        Returns:
            dict mapping servo_id to raw position
        """
        return {sid: servo.get_raw_position() for sid, servo in self.servos.items()}

    def get_statistics(self) -> dict[str, any]:
        """Get performance statistics."""
        time_since_last_read = time.perf_counter() - self._last_successful_read
        return {
            'read_count': self._read_count,
            'error_count': self._error_count,
            'time_since_last_read': time_since_last_read,
            'bus_stats': self.bus.get_statistics()
        }

    def close(self) -> None:
        """Shutdown leader arm and close serial connection."""
        logger.info("Closing leader arm...")
        self._stop_event.set()
        if self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)
        
        try:
            self.serial.close()
        except:
            pass
        
        logger.info("Leader arm closed")


# ============================================================================
# Follower Arm - Read and Write
# ============================================================================

class FollowerArm:
    """Follower arm interface with position control and monitoring.

    The follower arm can both read and write positions. It maintains
    a background thread for monitoring current positions and uses
    sync write for efficient position updates.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        update_rate: float = 100.0,
        enable_monitoring: bool = True,
        use_sync_write: bool = True,
        reconnect_on_error: bool = True
    ):
        """
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' or 'COM5')
            baudrate: Serial baud rate (default 1000000)
            update_rate: Monitoring update frequency in Hz
            enable_monitoring: Enable background position monitoring
            use_sync_write: Use sync write for batched position updates
            reconnect_on_error: Attempt to reconnect on serial errors
        """
        self.port = port
        self.baudrate = baudrate
        self.update_interval = 1.0 / update_rate
        self.enable_monitoring = enable_monitoring
        self.use_sync_write = use_sync_write
        self.reconnect_on_error = reconnect_on_error

        # Initialize serial connection
        self._connect()

        # Create servo objects with filtering for smoother motion
        self.servos = {
            sid: Servo(sid, JOINT_NAMES.get(sid, f"id{sid}"), use_filter=True)
            for sid in SERVO_IDS
        }

        # Thread control
        self._stop_event = threading.Event()
        self._monitor_thread = None

        # Statistics
        self._write_count = 0
        self._error_count = 0

        # Initial read and safe torque enable
        self._read_all_positions()
        self._enable_torque_all()

        # Start monitoring if enabled
        if self.enable_monitoring:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="FollowerMonitorThread"
            )
            self._monitor_thread.start()

        logger.info(f"Follower arm initialized on {port}")

    def _connect(self) -> None:
        """Establish serial connection."""
        try:
            self.serial = serial.Serial(
                self.port,
                baudrate=self.baudrate,
                timeout=0.005,
                write_timeout=0.01
            )
            time.sleep(0.05)
            self.serial.reset_input_buffer()
            self.bus = FeetechServobus(self.serial, max_retries=1)
            logger.info(f"Connected to follower arm on {self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to follower arm: {e}")
            raise

    def _reconnect(self) -> bool:
        """Attempt to reconnect after error."""
        try:
            logger.warning("Attempting to reconnect follower arm...")
            self.serial.close()
            time.sleep(0.5)
            self._connect()
            self._enable_torque_all()
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    def _read_all_positions(self) -> bool:
        """Read positions from all servos."""
        try:
            positions = self.bus.read_positions_batch(SERVO_IDS)
            
            for servo_id, raw_pos in positions.items():
                if raw_pos is not None:
                    self.servos[servo_id].update_position(raw_pos, is_leader=False)
            
            return True
        except Exception as e:
            logger.error(f"Error reading follower positions: {e}")
            return False

    def _enable_torque_all(self) -> None:
        """Safely enable torque on all servos."""
        logger.info("Enabling torque on follower arm...")
        success = self.bus.enable_all_torque_safely(SERVO_IDS, speed=0)
        if success:
            logger.info("Torque enabled successfully")
        else:
            logger.warning("Some servos failed to enable torque")

    def _monitor_loop(self) -> None:
        """Background thread for monitoring current positions."""
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            
            self._read_all_positions()
            
            # Maintain update rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self.update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Follower monitor loop stopped")

    def get_positions(self) -> dict[str, float]:
        """Get current normalized positions."""
        return {servo.name: servo.get_position() for servo in self.servos.values()}

    def set_positions(self, normalized_targets: dict[str, float]) -> bool:
        """Set multiple normalized positions efficiently.

        Args:
            normalized_targets: dict mapping joint name to normalized position [0, 1]

        Returns:
            True if positions were set successfully
        """
        try:
            if self.use_sync_write:
                # Build position data for sync write
                position_data = {}
                for servo in self.servos.values():
                    if servo.name in normalized_targets:
                        target_norm = normalized_targets[servo.name]
                        target_ticks = denormalize_position(target_norm, servo.id, is_follower=True)
                        position_data[servo.id] = target_ticks
                
                if position_data:
                    success = self.bus.write_positions_sync(position_data)
                    if success:
                        self._write_count += 1
                    return success
            else:
                # Individual writes (slower but more compatible)
                for servo in self.servos.values():
                    if servo.name in normalized_targets:
                        target_norm = normalized_targets[servo.name]
                        target_ticks = denormalize_position(target_norm, servo.id, is_follower=True)
                        self.bus.set_goal_position(servo.id, target_ticks)
                self._write_count += 1
                return True

        except Exception as e:
            logger.error(f"Error setting positions: {e}")
            self._error_count += 1
            return False

    def set_position(self, joint_name: str, normalized_position: float) -> bool:
        """Set a single joint position.

        Args:
            joint_name: Name of joint (yaw, shoulder, etc.)
            normalized_position: Target position [0, 1]

        Returns:
            True if successful
        """
        return self.set_positions({joint_name: normalized_position})

    def get_statistics(self) -> dict[str, any]:
        """Get performance statistics."""
        return {
            'write_count': self._write_count,
            'error_count': self._error_count,
            'bus_stats': self.bus.get_statistics()
        }

    def close(self) -> None:
        """Shutdown follower arm and close serial connection."""
        logger.info("Closing follower arm...")
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        try:
            self.bus.disable_all_torque(SERVO_IDS)
        except:
            pass
        
        try:
            self.serial.close()
        except:
            pass
        
        logger.info("Follower arm closed")


# ============================================================================
# Teleoperation functions
# ============================================================================

def teleoperate(
    leader: LeaderArm,
    follower: FollowerArm,
    update_rate: float = 100.0,
    print_stats: bool = True,
    stats_interval: float = 5.0
) -> None:
    """Mirror leader arm movements to follower arm.

    This is the main teleoperation loop that continuously reads the leader
    positions and writes them to the follower.

    Args:
        leader: Leader arm instance
        follower: Follower arm instance
        update_rate: Control loop frequency in Hz
        print_stats: Print statistics periodically
        stats_interval: Interval for printing stats (seconds)

    Note:
        Press Ctrl+C to stop teleoperation
    """
    logger.info(f"Starting teleoperation at {update_rate} Hz")
    logger.info("Press Ctrl+C to stop")

    update_interval = 1.0 / update_rate
    last_stats_time = time.perf_counter()
    loop_count = 0

    try:
        while True:
            loop_start = time.perf_counter()

            # Read leader positions
            leader_positions = leader.get_positions()

            # Write to follower
            follower.set_positions(leader_positions)

            loop_count += 1

            # Print statistics periodically
            if print_stats and (time.perf_counter() - last_stats_time) >= stats_interval:
                actual_rate = loop_count / (time.perf_counter() - last_stats_time)
                logger.info(f"Teleoperation rate: {actual_rate:.1f} Hz")
                logger.info(f"Leader stats: {leader.get_statistics()}")
                logger.info(f"Follower stats: {follower.get_statistics()}")
                
                loop_count = 0
                last_stats_time = time.perf_counter()

            # Maintain update rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > update_interval * 1.5:
                logger.warning(f"Loop took {elapsed*1000:.1f}ms (target: {update_interval*1000:.1f}ms)")

    except KeyboardInterrupt:
        logger.info("Teleoperation stopped by user")
    except Exception as e:
        logger.error(f"Teleoperation error: {e}")
        raise


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    import sys

    # Default configuration - macOS ports
    LEADER_PORT = '/dev/tty.usbmodem5B3E1228011'
    FOLLOWER_PORT = '/dev/tty.usbmodem5B3E1188421'

    try:
        # Initialize arms
        leader = LeaderArm(LEADER_PORT, update_rate=100)
        follower = FollowerArm(FOLLOWER_PORT, update_rate=100, use_sync_write=True)

        # Run teleoperation
        teleoperate(leader, follower, update_rate=100, print_stats=True)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        # Clean shutdown
        if 'leader' in locals():
            leader.close()
        if 'follower' in locals():
            follower.close()
        logger.info("Shutdown complete")
