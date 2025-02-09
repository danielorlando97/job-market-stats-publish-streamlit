from pymongo import MongoClient
from pymongo.collection import Collection
from environs import Env

env = Env()
env.read_env()

MONGO_HOST = env("MONGO_HOST")
MONGO_USER = env("MONGO_USER")
MONGO_PASSWORD = env("MONGO_PASSWORD")


class DBMongoJobs:

    def __init__(self) -> None:
        uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}"
        client = MongoClient(uri)
        self._db = client["Jobs"]

    @property
    def offers(self) -> Collection:
        return self._db["Offers"]

    def get_all(self, filters=None, fields=None, orden=[("published_at", -1)]):
        filters = filters or {}
        fields = fields or [
            "id",
            "name",
            "body",
            "published_at",

            "min_salary",
            "max_salary",
            "currency",
            "salary_type",

            "experience",
            "seniority",
            "modality",
            "company",
            "countries",

            "portal",
        ]

        projection = {field: 1 for field in fields} if fields else None

        cursor = self.offers.find(filters, projection).sort(orden)
        return (item for item in cursor)

    def get_jobs(self, page=0, page_size=25, *args, **kwargs):
        cursor = self.get_all(*args, **kwargs)
        skip = page * page_size

        cursor = cursor.skip(skip).limit(page_size)
        return [item for item in cursor]
