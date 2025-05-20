from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import DocumentNotFoundException, CouchbaseException
from datasets import Dataset
import json
from couch_db.config import settings

class CouchbaseConnect:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CouchbaseConnect, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, host, user, password, bucket, document):
        # control initialization with pattern Singleton
        if self._initialized:
            return
        
        # parameters for server connection
        self.host = host
        self.user = user
        self.password = password
        self.bucket = bucket 
        self.document = document
        # parameters for cluster connetion
        self.cluster = None        
        self.collection = None
        self._initialized = True
        self.connect()

    # function to connect with couchbase instance
    def connect(self):
        try:
            # define cluster
            self.cluster = Cluster(self.host, ClusterOptions(
                    PasswordAuthenticator(self.user, self.password)))            
            # open bucket 
            bucket = self.cluster.bucket(self.bucket)
            self.collection = bucket.default_collection()
            # initialize document
            self.init_document()
            print("Connection with Couchbase is successfully!")

        except CouchbaseException as ex:
            print(f"Failure in connection with couchbase: {ex}")
            raise

    # function to initialize document
    def init_document(self):
        try:
            doc = self.collection.get(self.document)
            data = doc.content_as[dict]
            required_fields = ["model_name", "question", "answer", "time", "score"]

            for field in required_fields:
                if field not in data:
                    data[field] = []
                    self.collection.upsert(self.document, data)                    

        except DocumentNotFoundException:
            # create a new document with default structure
            default_data = {"model_name": [], "question": [], "answer": [], "time": [], "score": []}
            self.collection.upsert(self.document, default_data)
            print("Initialized new document with default structure!")

        except Exception as e:
            print(f"Error initializing document: {e}")
            raise

    # function to insert new document into collection
    def insert(self, model_name, question, answer, time, score):
        try:
            # get document            
            doc = self.collection.get(self.document)
            data = doc.content_as[dict]
            
            # append new row in data
            data["model_name"].append(str(model_name))
            data["question"].append(str(question))
            data["answer"].append(str(answer))
            data["time"].append(float(time))
            data["score"].append(float(score) if score is not None else None)
            
            # save the document in the collection
            self.collection.upsert(self.document, data)
            inserted = True
            print("Data inserted successfully!")
        
        except CouchbaseException as e:
            inserted = False
            print("Error of insertion in couchbase!")

        return inserted

    # function to read couchbase collection
    def read_document(self):
        try:
            # read data
            result = self.collection.get(self.document)
            # generate structure of dictionary
            data = result.content_as[dict]
            # convert couchbase data to python dataset
            dataset = Dataset.from_dict(data)

            return dataset
        
        except CouchbaseException as ex:
            print(f"Error to read couchbase document: {ex}")
            return None

    # function to read the last document
    def read_last_document(self):
        return self.read_document()[-1]

# settings for couchbase connection
couchbase_cnn = CouchbaseConnect(**settings["couchbase_config"])
