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
            print("Connection with Couchbase is successfully!")
        except CouchbaseException as ex:
            print(f"Failure in connection with couchbase: {ex}")
            raise


# settings for couchbase connection
couchbase_cnn = CouchbaseConnect(**settings["couchbase_config"])
