from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import DocumentNotFoundException, CouchbaseException
from datasets import Dataset
from datetime import datetime
from couch_db.config import settings

class CouchbaseExperimentManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CouchbaseExperimentManager, cls).__new__(cls)
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

            if not isinstance(data, dict):
                data = {}
                self.collectio.upsert(self.document, data)

        except DocumentNotFoundException:
            self.collection.upsert(self.document, {})
            print(f"Document '{self.document}' created successfully.")
        except CouchbaseException as ex:
            print(f"Error initializing base document: {ex}")

    # function to initialize experiment
    def init_experiment(self, url, question):
        try:
            # generation of timestamp for the experiment key            
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
            experiment_key = f"experiment_{timestamp}"

            # get base document
            doc = self.collection.get(self.document)
            data = doc.content_as[dict]

            default_data = {
                "url": url,
                "question": question,
                "date": timestamp,
                "model_name": [],
                "answer": [],
                "time": [],
                "score": []
            }

            data[experiment_key] = default_data
            self.collection.upsert(self.document, data)
            print(f"Experiment initialized with key: {experiment_key}")

            return experiment_key

        except CouchbaseException as ex:
            print(f"Failure in initializing experiment: {ex}")
            return None

    # function to insert details of experiment for each model
    def insert(self, experiment_key, model_name, answer, time, score):
        try:
            doc = self.collection.get(self.document)
            data = doc.content_as[dict]

            # check if document exists
            if experiment_key not in data:
                print(f"Experiment {experiment_key} does not found!")
                return False

            data[experiment_key]["model_name"].append(str(model_name))
            data[experiment_key]["answer"].append(str(answer))
            data[experiment_key]["time"].append(float(time))
            data[experiment_key]["score"].append(float(score) if score is not None else None)

            # save the document in the collection
            self.collection.upsert(self.document, data)
            inserted = True
            print(f"Data inserted successfully into {experiment_key}!")        

        except CouchbaseException as ex:
            inserted = False
            print(f"Error in insert the data into {experiment_key}: {ex}")

        return inserted
    
    # function to read couchbase collection
    def read_documents(self):
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
        try:
            # Read the base document
            result = self.collection.get(self.document)
            data = result.content_as[dict]
            # Get the last experiment key (assuming sorted by timestamp)
            if data:
                last_key = sorted(data.keys())[-1]
                return Dataset.from_dict(data[last_key])
            return None
        except CouchbaseException as ex:
            print(f"Error reading last experiment: {ex}")
            return None

# settings for couchbase connection
couchbase_data = CouchbaseExperimentManager(**settings["couchbase_bench"])
