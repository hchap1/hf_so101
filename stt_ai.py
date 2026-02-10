from cli import Controller
from stt import listen_until_silence
from open_ai_test import call_agent
from time import sleep

arm = Controller()

print("About to start listening...")
initial_prompt = listen_until_silence()
print("Done listening.")
print(initial_prompt)

action = call_agent(initial_prompt)

while True:
    arm.add(action)
    result = arm.run()
    if result == "": break
    action = call_agent(result)
