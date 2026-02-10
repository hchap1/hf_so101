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
        self.commands.append(command)

    def run(self):
        for string in self.commands:

            command = string.split(" ")[0]
            parameters = [float(x) for x in string.split(" ")[1:]]
            
            if command == "sleep":
                duration = parameters[0]
                sleep(duration)
            elif command == "get": print(self.arm.format_arm_state_for_ai())
            elif command == "set":
                _ = self.arm.set_position_cm(parameters[0], parameters[1], parameters[2])

if __name__ == "__main__":
    controller = Controller()
    arguments = " ".join(sys.argv[1:]).split(", ")
    for argument in arguments:
        controller.add(argument)
    controller.run()
