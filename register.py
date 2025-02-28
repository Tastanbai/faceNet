from fastapi import APIRouter, Depends, HTTPException, status, Form
from pydantic import BaseModel
from typing import List
import mysql.connector
from auth import get_current_user, hash_password
from enum import Enum
import json

router = APIRouter()

DB_CONFIG = {
    "host": "facenet.tabet-kitap.kz",
    "user": "fastapi_user",
    "password": "secure_password",
    "database": "face_db",
    "port": 3306
}

# ✅ Автоматическое определение API на основе зарегистрированных маршрутов
class AvailableAPIs(str, Enum):
    compare_face = "compare-face"
    compare_face_qr = "compare-face-qr"
    process_patient = "process-patient"
    process_patient_base64 = "process-patient-base64"
    get_face_data = "get-face-data"

class RegisterUserSchema(BaseModel):
    username: str
    password: str
    is_admin: bool = False
    allowed_api: List[AvailableAPIs]  # ✅ Здесь создается выпадающий список

@router.post("/register")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),  # По умолчанию обычный пользователь
    allowed_api: List[AvailableAPIs] = Form(...),  # ✅ Выпадающий список API
    current_user: dict = Depends(get_current_user)
):
    """ Регистрация пользователя с выбором доступных API """

    # ✅ Преобразуем API в JSON-строку перед сохранением в базу
    allowed_api_json = json.dumps([api.value for api in allowed_api])

    # Только администратор может добавлять пользователей
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Только администраторы могут добавлять пользователей")

    hashed_password = hash_password(password)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO users (username, password, is_admin, allowed_api) VALUES (%s, %s, %s, %s)", 
                       (username, hashed_password, is_admin, allowed_api_json))
        conn.commit()

        cursor.close()
        conn.close()

        return {"message": f"Пользователь {username} успешно зарегистрирован", "allowed_api": allowed_api}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
