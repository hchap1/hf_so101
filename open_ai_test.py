import os, sys

import cv2
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
)
from cli import Controller

class AI:

    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

        self._messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    "You are controlling a robotic arm via a command line interface (CLI). "
                    "Your first action should always be to run the 'help' command to see available commands. "
                    "Respond only in valid CLI syntax. Your replies are executed directly in the CLI, "
                    "so they must be precise and correct. "
                    "If your message ends with 'help' or 'get', you will receive the CLI output and must plan the next steps. "
                    "If your message does not end with 'help' or 'get', the task sequence is considered complete. "
                    "Your goal will always be provided in plain English. "
                    "Take time to THINK properly about what you are trying to achieve. "
                    "Never call help more than once."
                )
            )
        ]

    def call_agent(self, prompt: str):
        self._messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=self._messages,
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        self._messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=reply))
        return reply

    # ============================
    # 📷 One-off webcam vision call
    # ============================

    def ask_about_webcam_image(self, question: str) -> str:
        """
        Captures one image from the webcam and asks the AI a one-off question about it.
        This does NOT persist to conversation memory.
        """

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture image")

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # Encode image to base64 PNG
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        response = self.client.responses.create(
            model="gpt-4.1",
            input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}",
                        },
                    ],
                }]
        )


        return response.output_text

    def work_it_out(self, prompt: str):
        return self.ask_about_webcam_image(
                f"You are one of two AI's working together to control a robot arm. You are able to see, the other AI cannot. Here is what the arm has been asked to do: {prompt}. You must analyse the image and return a NEW prompt for the AI that controls the arm, telling it all the information it needs to complete its task. Remember that it has no input except for this prompt, so for example, if you are asked to 'pick up the apple', your job is to locate the apple relative to the arm in your vision, and then return something like 'pick up the object 15cm in front of you and 10cm to the left, etc. If the image is not relevant to the prompt or the prompt does not require vision, simply return the same prompt back. Be extremely careful about what you tell it to do, as it may cause damage if the values are wrong. You need to figure out which direction is forwards for the arm based on the image before making your prediction of how far the distances are for the values. Some prompts are more complex and require analysis of multiple parts of the image. IE, if you were told to put the screwdriver on the phone, you must guess at the position of the screwdriver and the phone. Be super explicit with positions'"
                "Also, if something is on the table then remind the AI that the object is at ground level. be incredibly descriptive to it"
        )
