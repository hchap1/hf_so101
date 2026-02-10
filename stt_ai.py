from cli import Controller
from stt import listen_until_silence
from open_ai_test import AI
from time import sleep

arm = Controller()
ai = AI()

print("About to start listening...")
initial_prompt = listen_until_silence()
print("Done listening.")
print(initial_prompt)

action = ai.call_agent(initial_prompt)

while True:
    arm.add(action)
    result = arm.run()

    print(f"Called {action}, received |{result}|")

    if result == "": break
    action = ai.call_agent(result)
