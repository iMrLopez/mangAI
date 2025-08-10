"""
LLM Module
Uses LLaVA and GPT4 models to describe what is happening in each frame
"""
import ollama
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re

class LLM_Vision:

    def __init__(self,model="gpt-4.1"):
        self.prompt = """
        You are an expert storyteller. Your task is to describe what is happening in the scene and the characters that appear. 
        Do not include additional information that is not in the image. Do not describe text included in the image or text bubbles.
        Consider the following information when describing the scene:
        - What is happening in the panel? What is the context?
        - How many characters are there? What are they doing? How they look (facial expressions, posture)?
        - Pretend you are telling a story to someone who cannot see the image.
        - Limit your answer to less than 150 words.

        Additional to this, get the emotion sensed in the image. This is the list of emotions that can be used: happy, sad,angry,nervous,excited,calm and sarcastic.

        The output must be given in a json variable with the following format. Use double quotes for the start and end of the key names and strings:
        {
          "description": "description according to the instructions provided"
          "emotion": "emotion sensed (happy,angry,excited, etc)" 
        }
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

"""
Uses OpenAI API to construct the story and describe what is happening
"""     
class LLM_Narrator:

    def __init__(self,model="gpt-4.1"):
        self.prompt = """
        Act as a storyteller. Your task is to analyze some sentences and reword them so they make sense and describe a continuous story. 
        The next steps must be followed:
        - Do not change the logic, character descriptions or add actions or characters that were not originally there. 
        - For the wording, use a style to narrate stories and keep the interest of the person reading the description. 
        - You will get a list of sentences enumerated by a hyphen (-) character. Analyze all the sentences but when changing the wording, keep the sentences separated by the hypen. 

        For example: 
        Based on a list of sentences received as follows
        - There is a girl and a cat. The girl is a kid and the cat is fluffy and white. They are playing together. 
        - A cat, a dog and a girl appear in the scene. They girl is holding the cat and looks scared. 
        - There is a dog smelling the cat and a kid holding the cat. The dog seems curious but the kid is concerned about her cat. 

        The sentences must be reworded to have logic and connection between while keeping the original sentences separated by a hyphen:
        - There is a little gir playing with his friend who is a fluffy and white cat. 
        - While they are playing, a dog surprises them and the girl concerned about her cat goes and protect it from the dog. 
        - The dog is curious so it is approaching them and tries to smell the cat that the little girl is holding.
        """
        self.model = model

        if "gpt" in self.model:
            load_dotenv()
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = ""

    def getOriginalDescription(self,listOfDescriptions):

        newStringSeparatedByHyphen = ""

        for list in listOfDescriptions:
            jsonFormat = self.convertToJson(list)
            description = jsonFormat['description']
            newStringSeparatedByHyphen = newStringSeparatedByHyphen + "- " + description + "\n"

        return newStringSeparatedByHyphen

    def convertToJson(self, stringJson):
        jsonFormat = re.sub(r"```(?:json)?", "", stringJson, flags=re.IGNORECASE).strip("` \n")
        jsonFormat = jsonFormat.replace("'description'",'"description"')
        jsonFormat = jsonFormat.replace("'emotion'",'"emotion"')
        try:
            return json.loads(jsonFormat)
        except json.JSONDecodeError:
            pass
        try:
            comma_fixed = re.sub(
            r'("\s*:\s*[^,}\]]+)\s*(")',
            r'\1, \2',
            jsonFormat
        )
            return json.loads(comma_fixed)
        except json.JSONDecodeError:
            pass

    def updateFrameDescription(self,originalDescription):

        response = self.client.responses.create(
            model=self.model,
            instructions=self.prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": f"Update the following description: {originalDescription}"}
                    ],
                }
            ],
        )
        return response.output_text
    
    def extract_hyphen_sentences(self,paragraph: str):
        """
        Extracts sentences starting with a hyphen (-) from a paragraph.
        Returns a list of cleaned sentences (without the leading hyphen and extra spaces).
        """
        sentences = []
        for line in paragraph.splitlines():
            line = line.strip()
            if line.startswith("-"):
                sentences.append(line.lstrip("-").strip())  # remove leading hyphen & spaces
        return sentences
    
    def frameScript(self,llmOriginalDescription,ocrText):
        scriptArray = []
        # First, process the llm descriptions and update the descriptions to have a logical sequence
        originalDesc = self.getOriginalDescription(llmOriginalDescription)
        updatedDesc = self.updateFrameDescription(originalDesc)
        updatedDescArray = self.extract_hyphen_sentences(updatedDesc)
        """ The script will be given in the following order
        - Description of Frame 1
        - OCR Text of Frame 1
        ....
        - Description of Frame N
        - OCR Text of Frame N

        """
        for i in range(len(llmOriginalDescription)):
            # narrator
            scriptArray.append({"role": "narrator", "description": updatedDescArray[i],"emotion": "calm"})
            # character
            frameDescriptionJson = self.convertToJson(llmOriginalDescription[i])
            scriptArray.append({"role": "character", "description": ocrText[i]["cleaned_text"],"emotion": frameDescriptionJson["emotion"]})
            counter = 0
        for script in scriptArray:
            counter = counter + 1
            print("******* Frame " + str(counter))
            print(script)
            print("\n")







