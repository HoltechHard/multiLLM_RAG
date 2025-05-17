import json
import os
from dotenv import load_dotenv

def get_model(model_name):    
    with open("config/models.json") as f:
        data = json.load(f)

        if model_name == "OpenAI":
            load_dotenv()   # load environment variables
            api_key = os.getenv("OPENAI_CHAT_API_KEY")

            if api_key is None:
                raise ValueError("API key for OpenAI not found ...")
            else:
                data[model_name]["api_key"] = api_key
    
    return data[model_name]

def list_models():
    with open("config/models.json") as f:
        data = json.load(f)
    keys_models = list(data.keys())

    return keys_models

"""
if __name__ == "__main__":
    print(get_model("Deepseek"))
    print(get_model("OpenAI"))
"""
