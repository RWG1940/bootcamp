import sys
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from logs import LOGGER


class MilvusHelper:
    """
    Milvus Helper
    """
    def __init__(self):
        try:
            self.collection = None
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{MILVUS_HOST} and PORT:{MILVUS_PORT}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def set_collection(self, collection_name):
        try:
            self.collection = Collection(name=collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to set collection in Milvus: {e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        # Return if Milvus has the collection
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to get collection info to Milvus: {e}")
            sys.exit(1)

    def create_collection(self, collection_name):
        # Create milvus collection if not exists
        try:
            field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
            field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector",
                                    dim=VECTOR_DIMENSION, is_primary=False)
            schema = CollectionSchema(fields=[field1, field2], description="collection description")
            self.collection = Collection(name=collection_name, schema=schema)
            LOGGER.debug(f"Create Milvus collection: {collection_name}")
            return "OK"
        except Exception as e:
            LOGGER.error(f"Failed create collection in Milvus: {e}")
            sys.exit(1)

    def insert(self, collection_name, vectors):
        # Batch insert vectors to milvus collection
        try:
            self.set_collection(collection_name)
            data = [vectors]
            mr = self.collection.insert(data)
            ids = mr.primary_keys
            LOGGER.debug(
                    f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to insert data to Milvus: {e}")
            sys.exit(1)

    def create_index(self, collection_name):
        # Create IVF_FLAT index on milvus collection
        try:
            self.set_collection(collection_name)
            default_index = {"metric_type": METRIC_TYPE, "index_type": "IVF_FLAT", "params": {"nlist": 2048}}
            status = self.collection.create_index(field_name="embedding", index_params=default_index)
            if not status.code:
                LOGGER.debug(
                    f"Successfully create index in collection:{collection_name} with param:{default_index}")
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
        # Delete Milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.drop()
            LOGGER.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            LOGGER.error(f"Failed to drop collection: {e}")
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k):
        # Search vector in milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.load()
            search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
            res = self.collection.search(vectors, anns_field="embedding", param=search_params, limit=top_k)
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search vectors in Milvus: {e}")
            sys.exit(1)

    def count(self, collection_name):
        # Get the number of milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.flush()
            num = self.collection.num_entities
            LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)

    def delete_entity_by_id(self, collection_name, id):
        # Delete entity by id from milvus collection
        try:
            self.set_collection(collection_name)
            expr = f'id in [{id}]'
            self.collection.delete(expr)
            LOGGER.debug(f"Successfully delete entity id:{id} from collection:{collection_name}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to delete entity from Milvus: {e}")
            return False
