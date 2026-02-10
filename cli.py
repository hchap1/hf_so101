# The goal of this file is to provide a documented CLI that an AGENT can manipulate.

from agent import ArmController
from time import sleep
import sys


class Controller:
    commands: list[str] = []
    arm: ArmController

    def __init__(self):
        self.arm = ArmController()

    def add(self, command: str):
        for command in command.split(", "): self.commands.append(command)
        print(self.commands)

    def run(self) -> str:

        output = ""

        for string in self.commands:

            command = string.split(" ")[0]
            parameters = [float(x) for x in string.split(" ")[1:]]
            
            if command == "sleep":
                duration = parameters[0]
                sleep(duration)
            elif command == "get": output += self.arm.format_arm_state_for_ai() + "\n"
            elif command == "set":
                _ = self.arm.set_position_cm(
                    parameters[0], parameters[1], parameters[2],
                    wrist_pitch_deg=parameters[3],
                    roll=parameters[4],
                    claw=parameters[5]
                )

            else:
                with open("help.txt", "r") as help:
                    output += help.read()
                
        self.commands.clear()
        return output
