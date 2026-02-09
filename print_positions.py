#!/usr/bin/env python3
"""
Simple script to print joint positions of both leader and follower arms.
Follower motors are disabled so you can move them freely.
"""

import sys
import time
from protocol import LeaderArm, FollowerArm

def print_positions(leader, follower):
    """Print positions of both arms in a readable format."""
    
    # Clear screen for better readability (optional)
    print("\033[2J\033[H", end="")  # ANSI escape codes to clear screen
    
    print("=" * 70)
    print("LEADER ARM POSITIONS (read-only)")
    print("=" * 70)
    
    leader_pos = leader.get_positions()
    for joint_name in ['yaw', 'shoulder', 'elbow', 'wrist', 'roll', 'claw']:
        if joint_name in leader_pos:
            normalized = leader_pos[joint_name]
            print(f"  {joint_name:>10s}: {normalized:7.4f} (normalized)")
    
    print("\n" + "=" * 70)
    print("FOLLOWER ARM POSITIONS (motors disabled - you can move it)")
    print("=" * 70)
    
    follower_pos = follower.get_positions()
    for joint_name in ['yaw', 'shoulder', 'elbow', 'wrist', 'roll', 'claw']:
        if joint_name in follower_pos:
            normalized = follower_pos[joint_name]
            print(f"  {joint_name:>10s}: {normalized:7.4f} (normalized)")
    
    print("\n" + "=" * 70)
    print("Press Ctrl+C to exit")
    print("=" * 70)


def main():
    """Main function to initialize arms and print positions."""
    
    # Default ports for macOS
    LEADER_PORT_DEFAULT = '/dev/tty.usbmodem5B3E1228011'
    FOLLOWER_PORT_DEFAULT = '/dev/tty.usbmodem5B3E1188421'
    
    # Get port names from user
    print("\nEnter serial port names for your arms")
    print(f"(Press Enter to use defaults)\n")
    
    leader_input = input(f"Leader arm port [{LEADER_PORT_DEFAULT}]: ").strip()
    leader_port = leader_input if leader_input else LEADER_PORT_DEFAULT
    
    follower_input = input(f"Follower arm port [{FOLLOWER_PORT_DEFAULT}]: ").strip()
    follower_port = follower_input if follower_input else FOLLOWER_PORT_DEFAULT
    
    print(f"\nUsing Leader: {leader_port}")
    print(f"Using Follower: {follower_port}\n")
    
    leader = None
    follower = None
    
    try:
        print("\nInitializing leader arm...")
        leader = LeaderArm(leader_port, update_rate=50)
        
        print("Initializing follower arm...")
        follower = FollowerArm(follower_port, update_rate=50)
        
        # Disable torque on follower so you can move it freely
        print("Disabling follower arm motors...")
        follower.bus.disable_all_torque([1, 2, 3, 4, 5, 6])
        time.sleep(0.1)
        
        print("\nReady! Positions will update automatically.\n")
        time.sleep(1)
        
        # Continuously print positions
        while True:
            print_positions(leader, follower)
            time.sleep(0.2)  # Update 5 times per second
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean shutdown
        if follower:
            print("Closing follower arm...")
            follower.close()
        if leader:
            print("Closing leader arm...")
            leader.close()
        print("Done!")


if __name__ == "__main__":
    main()
