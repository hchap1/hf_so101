#!/usr/bin/env python3
"""
Quick-start teleoperation script.
Simply run this to start mirroring leader to follower.
"""

import sys
from protocol import LeaderArm, FollowerArm, teleoperate

# Your ports
LEADER_PORT = '/dev/tty.usbmodem5B3E1228011'
FOLLOWER_PORT = '/dev/tty.usbmodem5B3E1188421'

def main():
    """Quick start teleoperation."""
    
    print("=" * 60)
    print("SO-101 Teleoperation - Quick Start")
    print("=" * 60)
    print(f"\nLeader:   {LEADER_PORT}")
    print(f"Follower: {FOLLOWER_PORT}")
    print("\nInitializing...")
    
    leader = None
    follower = None
    
    try:
        leader = LeaderArm(LEADER_PORT, update_rate=100)
        print("✓ Leader arm ready")
        
        follower = FollowerArm(FOLLOWER_PORT, update_rate=100, use_sync_write=True)
        print("✓ Follower arm ready")
        
        print("\n" + "=" * 60)
        print("Starting teleoperation at 100 Hz")
        print("Move the leader arm and the follower will mirror it")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        teleoperate(leader, follower, update_rate=100, print_stats=True)
        
    except KeyboardInterrupt:
        print("\n\nStopping teleoperation...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if leader:
            print("Closing leader arm...")
            leader.close()
        if follower:
            print("Closing follower arm...")
            follower.close()
        print("Done!")


if __name__ == "__main__":
    main()
