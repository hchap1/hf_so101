#!/usr/bin/env python3
"""
Diagnostic script to check raw positions and help calibrate the arms.
This will show you the RAW encoder values to help debug normalization.
"""

import sys
import time
from feetech_servo_protocol import FeetechServobus, LEADER_GEAR_RATIOS, FOLLOWER_GEAR_RATIOS
import serial

def print_diagnostic(leader_bus, follower_bus):
    """Print detailed diagnostic information."""
    
    # Clear screen
    print("\033[2J\033[H", end="")
    
    print("=" * 100)
    print("DIAGNOSTIC MODE - RAW ENCODER VALUES")
    print("=" * 100)
    print()
    
    # Read leader positions
    print("LEADER ARM (Raw Encoder Ticks)")
    print("-" * 100)
    leader_positions = leader_bus.read_positions_batch([1, 2, 3, 4, 5, 6])
    
    joint_names = {1: 'yaw', 2: 'shoulder', 3: 'elbow', 4: 'wrist', 5: 'roll', 6: 'claw'}
    
    for servo_id in [1, 2, 3, 4, 5, 6]:
        name = joint_names[servo_id]
        raw = leader_positions.get(servo_id, None)
        gear_ratio = LEADER_GEAR_RATIOS[servo_id]
        
        if raw is not None:
            normalized = raw / 4096.0  # Correct normalization
            print(f"  ID {servo_id} ({name:>8s}): Raw={raw:5d}  GearRatio={gear_ratio:3d}  Normalized={normalized:8.4f}")
        else:
            print(f"  ID {servo_id} ({name:>8s}): READ FAILED")
    
    print()
    
    # Read follower positions
    print("FOLLOWER ARM (Raw Encoder Ticks)")
    print("-" * 100)
    follower_positions = follower_bus.read_positions_batch([1, 2, 3, 4, 5, 6])
    
    for servo_id in [1, 2, 3, 4, 5, 6]:
        name = joint_names[servo_id]
        raw = follower_positions.get(servo_id, None)
        gear_ratio = FOLLOWER_GEAR_RATIOS[servo_id]
        
        if raw is not None:
            normalized = raw / 4096.0  # Correct normalization
            print(f"  ID {servo_id} ({name:>8s}): Raw={raw:5d}  GearRatio={gear_ratio:3d}  Normalized={normalized:8.4f}")
        else:
            print(f"  ID {servo_id} ({name:>8s}): READ FAILED")
    
    print()
    print("=" * 100)
    print("COMPARISON (when arms are in same position, these should match)")
    print("=" * 100)
    
    for servo_id in [1, 2, 3, 4, 5, 6]:
        name = joint_names[servo_id]
        leader_raw = leader_positions.get(servo_id, 0)
        follower_raw = follower_positions.get(servo_id, 0)
        
        leader_norm = leader_raw / 4096.0 if leader_raw else 0
        follower_norm = follower_raw / 4096.0 if follower_raw else 0
        
        diff = abs(leader_norm - follower_norm)
        status = "✓ MATCH" if diff < 0.05 else "✗ MISMATCH"  # 5% tolerance
        
        print(f"  {name:>8s}: Leader={leader_norm:8.4f}  Follower={follower_norm:8.4f}  Diff={diff:7.4f}  {status}")
    
    print()
    print("=" * 100)
    print("Press Ctrl+C to exit")
    print("=" * 100)


def main():
    """Main diagnostic routine."""
    
    # Default ports for macOS
    LEADER_PORT_DEFAULT = '/dev/tty.usbmodem5B3E1228011'
    FOLLOWER_PORT_DEFAULT = '/dev/tty.usbmodem5B3E1188421'
    
    print("\nDiagnostic Mode - Raw Encoder Analysis")
    print("This will show you the RAW encoder values to debug normalization issues.\n")
    
    leader_input = input(f"Leader arm port [{LEADER_PORT_DEFAULT}]: ").strip()
    leader_port = leader_input if leader_input else LEADER_PORT_DEFAULT
    
    follower_input = input(f"Follower arm port [{FOLLOWER_PORT_DEFAULT}]: ").strip()
    follower_port = follower_input if follower_input else FOLLOWER_PORT_DEFAULT
    
    print(f"\nUsing Leader: {leader_port}")
    print(f"Using Follower: {follower_port}\n")
    
    leader_ser = None
    follower_ser = None
    
    try:
        print("\nConnecting to leader arm...")
        leader_ser = serial.Serial(leader_port, baudrate=1_000_000, timeout=0.01)
        time.sleep(0.05)
        leader_ser.reset_input_buffer()
        leader_bus = FeetechServobus(leader_ser)
        
        print("Connecting to follower arm...")
        follower_ser = serial.Serial(follower_port, baudrate=1_000_000, timeout=0.01)
        time.sleep(0.05)
        follower_ser.reset_input_buffer()
        follower_bus = FeetechServobus(follower_ser)
        
        # Disable follower motors
        print("Disabling follower motors...")
        follower_bus.disable_all_torque([1, 2, 3, 4, 5, 6])
        time.sleep(0.1)
        
        print("\nReady! Move both arms to the SAME position and compare values.\n")
        time.sleep(2)
        
        # Continuously update
        while True:
            print_diagnostic(leader_bus, follower_bus)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        if follower_ser:
            follower_ser.close()
        if leader_ser:
            leader_ser.close()
        print("Done!")


if __name__ == "__main__":
    main()
