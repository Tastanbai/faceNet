# patients.py
from fastapi import APIRouter, Request, HTTPException, Depends
from models import FaceData
from auth import get_current_user, check_permission

router = APIRouter()

@router.delete("/patients/{patient_id}")
async def delete_patient(patient_id: str, request: Request, user: dict = Depends(get_current_user)):
    await check_permission(user, request)
    result = await FaceData.filter(patient_id=patient_id, is_delete=False).update(is_delete=True)
    if result:
        return {"message": "Пациент удалён (soft delete)."}
    raise HTTPException(status_code=404, detail="Пациент не найден или уже удалён.")
