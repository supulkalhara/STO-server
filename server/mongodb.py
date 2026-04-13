"""
MongoDB async client via Motor.

Set MONGODB_URL env var to your Atlas connection string:
  mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority

Falls back to a local MongoDB instance for development if the env var is absent.
"""
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None

MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME: str = os.getenv("MONGODB_DB", "safetakeoff")


def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        logger.info("MongoDB client created (db=%s)", DB_NAME)
    return _client


def get_mongo_db() -> AsyncIOMotorDatabase:
    global _db
    if _db is None:
        _db = get_mongo_client()[DB_NAME]
    return _db


async def close_mongo():
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")
