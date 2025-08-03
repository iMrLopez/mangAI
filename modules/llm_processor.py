"""
LLM Module
Uses LLaVA and GPT4 models to describe what is happening in each frame
"""
import ollama
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv

class LLM_Vision:

    def __init__(self,model="gpt-4.1"):
        self.prompt = """
        You are an expert storyteller. Your task is to describe what is happening in the scene and the characters that appear. 
        Do not include additional information that is not in the image. Do not describe text included in the image or text bubbles.
        Consider the following information when describing the scene:
        - What is happening in the panel? What is the emotional tone or context?
        - How many characters are there? What are they doing? How they look (facial expressions, posture)?
        - Pretend you are telling a story to someone who cannot see the image.
        - Limit your answer to less than 150 words.
        """
        self.model = model

        if "gpt" in self.model:
            load_dotenv()
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = ""
    def updateModel(self, newModel):
        self.model = newModel

    def updatePrompt(self, newPrompt):
        self.prompt = newPrompt

    def processImages(self, imgPath):
        # Send request
        if self.model == 'llava':
            response = self.llavaRequests(imgPath)
        elif "gpt" in self.model:
            response = self.gptRequest(imgPath)
        else:
            response = 'Invalid model selected'
        return response
    
    def llavaRequests(self, imgPath):
        response = ollama.chat(
            model = self.model,
            messages=[
                {
                    'role': 'system',
                    'content': self.prompt
                },
                {
                    'role': 'user',
                    'content': 'Describe this image in less than 50 words. Do not describe text included in the image or text bubbles',
                    'images': [imgPath]
                },
            ],
            options = {
                'temperature': 0.0
            }
        )
        return response['message']['content']
    
    def gptRequest(self, imgPath):
        # Encode image first
        base64_image = self.encode_image(imgPath)
        response = self.client.responses.create(
            model=self.model,
            instructions=self.prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": "Describe the following image" },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
        return response.output_text
        

    def encode_image(self, imgPath):
        with open(imgPath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

