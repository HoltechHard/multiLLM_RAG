import json

def get_model(model_name):
    with open("config/models.json") as f:
        data = json.load(f)
    
    return data[model_name]

def list_models():
    with open("config/models.json") as f:
        data = json.load(f)
    keys_models = list(data.keys())

    return keys_models

"""
if __name__ == "__main__":
    print(get_model("Deepseek"))
"""
