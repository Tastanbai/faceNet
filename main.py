import os
import uuid
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Query
from models import FaceNet, FaceData, QR
from fastapi.responses import JSONResponse
from starlette_exporter import PrometheusMiddleware, handle_metrics  
from face_processing import detector, embedder

# Импорт общих функций для работы с изображениями и эмбеддингами
from face_processing import (
    load_image_from_upload,
    load_image_from_base64,
    detect_face,
    get_embedding,
    calculate_hash,
    save_face_and_embedding  # уже реализована через asyncio.to_thread
)
# Импорт обёртки для Faiss
from faiss_wrapper import FaissIndexWrapper
# Импорт функций авторизации/проверки прав (реализованы в auth.py)
from auth import get_current_user, check_permission

# Импорт моделей и регистрации Tortoise ORM
from tortoise import fields, models, Tortoise
from tortoise.contrib.fastapi import register_tortoise

# Ваши роутеры авторизации/регистрации (если есть)
from auth import router as auth_router, get_current_user
from register import router as register_router
from patients import router as patients_router


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(auth_router)
app.include_router(register_router)
app.include_router(patients_router)

# Добавляем Prometheus-мидлвер для сбора метрик
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

# Глобальный список для хранения эмбеддингов в памяти: (patient_id, hospital_id, embedding, emb_path)
EMBEDDINGS_IN_MEMORY: List[tuple] = []

# Инициализируем обёртку для Faiss-индекса
faiss_index_wrapper = FaissIndexWrapper()

# Регистрируем Tortoise ORM (используем MySQL драйвер aiomysql)
from tortoise.contrib.fastapi import register_tortoise

register_tortoise(
    app,
    db_url="mysql://fastapi_user:secure_password@facenet.tabet-kitap.kz:3306/face_db",
    modules={"models": ["models"]},  # "models" – это имя файла без расширения
    generate_schemas=False,
    add_exception_handlers=True,
)

# Глобальный обработчик ошибок – логирует исключения
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Ошибка в запросе {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера. Попробуйте позже."},
    )

@app.on_event("startup")
async def on_startup():
    try:

        # Прогрев MTCNN (детекция лица)
        dummy_face = np.random.rand(160, 160, 3).astype("uint8")
        _ = detector.detect_faces(dummy_face)  # Холостой прогон

        # Прогрев модели FaceNet
        dummy_embedding = np.random.rand(160, 160, 3).astype("float32")
        _ = embedder.embeddings([dummy_embedding])

        # Загружаем эмбеддинги из БД и строим Faiss-индекс
        await load_embeddings_on_startup()
        logger.info("Приложение запущено.")
    except Exception as e:
        logger.error(f"Ошибка при старте: {e}")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Приложение останавливается.")

async def load_embeddings_on_startup():
    try:
        # Загружаем все записи из таблицы FaceNet
        records = await FaceNet.all().values("patient_id", "hospital_id", "emb_path")
        temp_list = []
        for rec in records:
            emb_path = rec["emb_path"]
            if os.path.exists(emb_path):
                embedding = np.load(emb_path)
                temp_list.append((rec["patient_id"], rec["hospital_id"], embedding, emb_path))
        global EMBEDDINGS_IN_MEMORY
        EMBEDDINGS_IN_MEMORY = temp_list

        # Строим Faiss-индекс на основе загруженных эмбеддингов
        faiss_index_wrapper.build_index(EMBEDDINGS_IN_MEMORY)
        total = faiss_index_wrapper.index.ntotal if faiss_index_wrapper.index else 0
        logger.info(f"[startup] Загружено {len(EMBEDDINGS_IN_MEMORY)} эмбеддингов, индекс Faiss содержит {total} эмбеддингов.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке эмбеддингов: {e}")

@app.post("/process-patient/")
async def process_patient(
    request: Request,
    patient_id: str = Form(...),
    hospital_id: str = Form(...),
    branch_id: str = Form(...),
    palata_id: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    await check_permission(user, request)
    start_time = time.perf_counter()

    # 1. Загрузка изображения, детекция лица, вычисление эмбеддинга и хэширование
    img = await load_image_from_upload(file)
    face_img = await detect_face(img)
    embedding = await get_embedding(face_img)
    emb_hash = calculate_hash(embedding)

    # 2. Проверка дубликата по emb_hash
    duplicate = await FaceNet.filter(emb_hash=emb_hash).first()
    if duplicate:
        return JSONResponse(content={"status": False, "message": "Это лицо уже сохранено."})

    # 3. Асинхронное сохранение файлов
    image_filename, embedding_filename = await save_face_and_embedding(face_img, embedding)

    # 4. Запись в БД через ORM
    created_at_value = datetime.now()
    await FaceNet.create(
        patient_id=patient_id,
        hospital_id=hospital_id,
        branch_id=branch_id,
        palata_id=palata_id,
        image_path=image_filename,
        emb_path=embedding_filename,
        emb_hash=emb_hash,
    )

    # 5. Обновление кэша эмбеддингов и Faiss-индекса
    EMBEDDINGS_IN_MEMORY.append((patient_id, hospital_id, embedding, embedding_filename))
    faiss_index_wrapper.add_embedding(patient_id, hospital_id, embedding)

    elapsed_time = time.perf_counter() - start_time
    return JSONResponse(content={
        "status": True,
        "message": "Эмбеддинг успешно сохранён.",
        "store_time": elapsed_time,
        "created_at": created_at_value.isoformat()
    })

@app.post("/process-patient-base64/")
async def process_patient_base64(
    request: Request,
    patient_id: str = Form(...),
    hospital_id: str = Form(...),
    branch_id: str = Form(...),
    palata_id: str = Form(...),
    file_base64: str = Form(...),
    user: dict = Depends(get_current_user)
):
    await check_permission(user, request)
    start_time = time.perf_counter()

    try:
        img = await load_image_from_base64(file_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при декодировании base64: {e}")

    face_img = await detect_face(img)
    embedding = await get_embedding(face_img)
    emb_hash = calculate_hash(embedding)

    duplicate = await FaceNet.filter(emb_hash=emb_hash).first()
    if duplicate:
        return JSONResponse(content={"status": False, "message": "Это лицо уже сохранено."})

    image_filename, embedding_filename = await save_face_and_embedding(face_img, embedding)
    created_at_value = datetime.now()
    await FaceNet.create(
        patient_id=patient_id,
        hospital_id=hospital_id,
        branch_id=branch_id,
        palata_id=palata_id,
        image_path=image_filename,
        emb_path=embedding_filename,
        emb_hash=emb_hash,
    )
    EMBEDDINGS_IN_MEMORY.append((patient_id, hospital_id, embedding, embedding_filename))
    faiss_index_wrapper.add_embedding(patient_id, hospital_id, embedding)

    elapsed_time = time.perf_counter() - start_time
    return JSONResponse(content={
        "status": True,
        "message": "Эмбеддинг успешно сохранён (Base64).",
        "store_time": elapsed_time,
        "created_at": created_at_value.isoformat()
    })

@app.post("/compare-face/")
async def compare_face(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    await check_permission(user, request)
    start_time = time.perf_counter()

    img = await load_image_from_upload(file)
    face_img = await detect_face(img)
    query_embedding = await get_embedding(face_img)

    k = 1
    try:
        distances, indices = faiss_index_wrapper.search(query_embedding, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске в Faiss: {e}")

    best_similarity = float(distances[0][0])
    best_index = indices[0][0]
    if best_index >= len(faiss_index_wrapper.metadata):
        raise HTTPException(status_code=500, detail="Некорректный индекс в Faiss.")

    matched_patient_id, matched_hospital_id = faiss_index_wrapper.metadata[best_index]
    match_percentage = best_similarity * 100.0
    compare_time = time.perf_counter() - start_time
    status = best_similarity >= 0.7

    await FaceData.create(
        patient_id=matched_patient_id,
        hospital_id=matched_hospital_id,
        status=status,
        similarity_percentage=match_percentage,
        comparison_time=compare_time,
    )

    if status:
        return JSONResponse(content={
            "status": True,
            "match_percentage": match_percentage,
            "compare_time": compare_time,
            "matched_patient_id": matched_patient_id,
            "matched_hospital_id": matched_hospital_id
        })
    else:
        return JSONResponse(content={
            "status": False,
            "message": "Совпадений не найдено.",
            "match_percentage": match_percentage,
            "compare_time": compare_time,
            "matched_patient_id": None,
            "matched_hospital_id": None
        })

@app.post("/compare-face-qr/")
async def compare_face_qr(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    await check_permission(user, request)
    start_time = time.perf_counter()

    img = await load_image_from_upload(file)
    face_img = await detect_face(img)
    query_embedding = await get_embedding(face_img)

    try:
        distances, indices = faiss_index_wrapper.search(query_embedding, k=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске в Faiss: {e}")
    
    best_similarity = float(distances[0][0])
    best_index = indices[0][0]
    if best_index >= len(faiss_index_wrapper.metadata):
        raise HTTPException(status_code=500, detail="Некорректный индекс в Faiss.")

    matched_patient_id, _ = faiss_index_wrapper.metadata[best_index]
    matched = best_similarity >= 0.7

    if matched and matched_patient_id is not None:
        await QR.create(status=True, patient_id=matched_patient_id)
        return JSONResponse(content={
            "status": True,
            "patient_id": matched_patient_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        return JSONResponse(content={
            "status": False,
            "patient_id": None,
            "created_at": None
        })
    
# Функция валидации даты
def validate_date(date_str: Optional[str]) -> Optional[str]:
    if date_str:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Неверный формат даты: {date_str}. Используйте YYYY-MM-DD"
            )
    return None

@app.post("/get-face-data/")
async def get_face_data(
    request: Request,
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    hospital_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    await check_permission(user, request)

    # Валидация даты
    start_date = validate_date(start_date)
    end_date = validate_date(end_date)

    query_filters = {}
    if hospital_id:
        query_filters["hospital_id"] = hospital_id
    if patient_id:
        query_filters["patient_id"] = patient_id
    if start_date and end_date:
        query_filters["timestamp__range"] = [start_date + " 00:00:00", end_date + " 23:59:59"]
    elif start_date:
        query_filters["timestamp__gte"] = start_date + " 00:00:00"
    elif end_date:
        query_filters["timestamp__lte"] = end_date + " 23:59:59"

    records = await FaceData.filter(**query_filters).order_by("-timestamp").values()
    return {"count": len(records), "data": records}
