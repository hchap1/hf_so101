import os, sys
from openai import OpenAI
from openai import chat
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam
from cli import Controller

# Read API key from environment
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

# In-memory conversation state (persists while program runs)
_messages = [
    ChatCompletionUserMessageParam(
        role="user",
        content=(
            "You are controlling a robotic arm via a command line interface (CLI). "
            "Your first action should always be to run the 'help' command to see available commands. "
            "Respond only in valid CLI syntax. Your replies are executed directly in the CLI, "
            "so they must be precise and correct. "
            "If your message ends with 'help' or 'get', you will receive the CLI output and must plan the next steps. "
            "If your message does not end with 'help' or 'get', the task sequence is considered complete. "
            "Your goal will always be provided in plain English."
            "Take time to THINK properly about you are trying to achieve. For example, if you are told to move something you must break it down into the smaller steps: Go above it -> lower down -> close claw without moving -> move to target position -> open the claw at the target position"
        )
    )
]

def call_agent(prompt: str):
    global _messages

    _messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=_messages,
        temperature=0.7,
    )

    reply = response.choices[0].message.content

    _messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=reply))

    return reply
