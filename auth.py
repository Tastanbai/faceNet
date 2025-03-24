# auth.py
import json
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import os

from models import User  # <-- добавляем
from tortoise.exceptions import DoesNotExist  # <-- для отлова
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Через Tortoise ищем пользователя
    user = await User.filter(username=form_data.username).first()
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Создаём токен
    access_token = create_access_token(data={
        "sub": user.username,
        "is_admin": user.is_admin,
        "allowed_api": user.allowed_api  # Это может быть JSON-строка
    })
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        is_admin = payload.get("is_admin", False)
        allowed_api = payload.get("allowed_api", "[]")

        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Недействительный токен")

        return {"username": username, "is_admin": is_admin, "allowed_api": allowed_api}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Токен истек")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Недействительный токен")

# Проверка прав – без изменений
async def check_permission(user: dict, request: Request):
    allowed_api_str = user.get("allowed_api", "[]")
    if isinstance(allowed_api_str, str):
        try:
            allowed_api = json.loads(allowed_api_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Ошибка в формате allowed_api")
    elif isinstance(allowed_api_str, list):
        allowed_api = allowed_api_str
    else:
        raise HTTPException(status_code=500, detail="Некорректный формат allowed_api")

    api_name = request.scope["path"].strip("/")
    allowed_api = [api.strip("/") for api in allowed_api]

    if "*" in allowed_api or api_name in allowed_api:
        return
    raise HTTPException(status_code=403, detail=f"Нет доступа к API {api_name}")
