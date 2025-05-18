import json

def read_dbconfig():
    with open("couch_db/config.json", "r") as file:
        data = json.load(file)
    
    return data

settings = read_dbconfig()
