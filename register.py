# register.py
from fastapi import APIRouter, Depends, HTTPException, status, Form
from pydantic import BaseModel
from typing import List
from enum import Enum
import json
import os

from auth import get_current_user, hash_password
from models import User  # <-- наша ORM-модель
from tortoise.exceptions import IntegrityError

router = APIRouter()

class AvailableAPIs(str, Enum):
    compare_face = "compare-face"
    compare_face_qr = "compare-face-qr"
    process_patient = "process-patient"
    process_patient_base64 = "process-patient-base64"
    get_face_data = "get-face-data"

@router.post("/register")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
    allowed_api: List[AvailableAPIs] = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """ Регистрация пользователя с выбором доступных API """
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Только администраторы могут добавлять пользователей")

    hashed_password = hash_password(password)
    allowed_api_str = json.dumps([api.value for api in allowed_api])

    try:
        new_user = await User.create(
            username=username,
            password=hashed_password,
            is_admin=is_admin,
            allowed_api=allowed_api_str
        )
        return {"message": f"Пользователь {username} успешно зарегистрирован", "allowed_api": allowed_api}
    except IntegrityError as e:
        # Например, если username уникален и уже есть
        raise HTTPException(status_code=400, detail=f"Ошибка: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
