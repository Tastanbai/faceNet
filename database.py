import os
import aiomysql
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()  

host = os.getenv("DB_HOST")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_DATABASE")
port = int(os.getenv("DB_PORT"))

aiomysql_pool = None

async def init_db_pool():
    global aiomysql_pool
    aiomysql_pool = await aiomysql.create_pool(
        host=host,
        port=port,
        user=user,
        password=password,
        db=database,
        autocommit=True,
        minsize=1,
        maxsize=10
    )


async def get_db_connection():
    global aiomysql_pool
    if aiomysql_pool is None:
        raise HTTPException(status_code=500, detail="Пул соединений не инициализирован")
    return await aiomysql_pool.acquire()

async def release_db_connection(conn):
    global aiomysql_pool
    if aiomysql_pool and conn:
        aiomysql_pool.release(conn)
