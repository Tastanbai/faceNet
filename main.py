# import time
# import os
# import uuid
# import hashlib
# import cv2
# import numpy as np
# import aiomysql
# from mtcnn import MTCNN
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
# from fastapi.responses import JSONResponse
# from keras_facenet import FaceNet
# from datetime import datetime


# import base64
# import mysql.connector
# from io import BytesIO
# import time
# from datetime import datetime
# import os
# from auth import router as auth_router, get_current_user
# from register import router as register_router
# from fastapi import Request
# import json
# from typing import List, Optional

# app = FastAPI()

# app.include_router(auth_router)
# app.include_router(register_router)

# # Конфигурация подключения к БД
# DB_CONFIG = {
#     "host": "face.tabet-kitap.kz",
#     "user": "fastapi_user",
#     "password": "secure_password",
#     "database": "face_db",
#     "port": 3306
# }

# # Создаём директории для хранения изображений и эмбеддингов (если их ещё нет)
# os.makedirs("images", exist_ok=True)
# os.makedirs("embeddings", exist_ok=True)

# # Инициализируем детектор лиц и модель для эмбеддингов
# detector = MTCNN()
# embedder = FaceNet()

# # Порог схожести для определения совпадения
# SIMILARITY_THRESHOLD = 0.7


# async def check_permission(user: dict, request):
#     """ Проверяет доступ пользователя к API """
    
#     allowed_api_str = user.get("allowed_api", "[]")  # Получаем строку
#     try:
#         allowed_api = json.loads(allowed_api_str)  # Конвертируем JSON в список
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Ошибка в формате данных API")

#     api_name = request.scope["path"].strip("/")  

#     if "*" in allowed_api or api_name in allowed_api:
#         return  

#     raise HTTPException(status_code=403, detail=f"У вас нет доступа к API {api_name}")


# def cosine_similarity(a, b):
#     """Вычисление косинусной схожести между двумя векторами."""
#     a_norm = np.linalg.norm(a)
#     b_norm = np.linalg.norm(b)
#     if a_norm == 0 or b_norm == 0:
#         return 0.0
#     return np.dot(a, b) / (a_norm * b_norm)

# def calculate_hash(embedding):
#     """Вычисляет SHA256-хэш от эмбеддинга"""
#     return hashlib.sha256(embedding.tobytes()).hexdigest()

# async def get_db_connection():
#     """Асинхронное получение подключения к базе данных."""
#     conn = await aiomysql.connect(
#         host=DB_CONFIG["host"],
#         port=DB_CONFIG["port"],
#         user=DB_CONFIG["user"],
#         password=DB_CONFIG["password"],
#         db=DB_CONFIG["database"],
#         autocommit=True
#     )
#     return conn

# @app.post("/process-patient")
# async def process_patient(
#     request: Request,
#     patient_id: str = Form(...),
#     hospital_id: str = Form(...),
#     branch_id: str = Form(...),
#     palata_id: str = Form(...),
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
# ):
#     await check_permission(user, request)
    
#     """
#     Принимает изображение, обнаруживает лицо, вычисляет эмбеддинг и сохраняет его в БД.
#     """
#     start_time = time.perf_counter()

#     # Читаем изображение
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

#     # Обнаружение лица
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

#     face_data = detection_results[0]
#     x, y, width, height = face_data['box']
#     face_img = img[y:y+height, x:x+width]

#     # Вычисляем эмбеддинг
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     embedding = embeddings[0]

#     # Вычисляем хэш эмбеддинга
#     emb_hash = calculate_hash(embedding)

#     # Проверяем, есть ли уже такой эмбеддинг в базе
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor() as cursor:
#             await cursor.execute("SELECT id FROM faceNet WHERE emb_hash = %s", (emb_hash,))
#             duplicate = await cursor.fetchone()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при запросе к БД: {str(e)}")
#     finally:
#         conn.close()

#     if duplicate:
#         return JSONResponse(content={
#             "status": False,
#             "message": "Это лицо уже сохранено."
#         })

#     # Сохраняем файлы
#     image_filename = f"images/{uuid.uuid4()}.jpg"
#     embedding_filename = f"embeddings/{uuid.uuid4()}.npy"

#     cv2.imwrite(image_filename, face_img)
#     np.save(embedding_filename, embedding)

#     # Добавляем запись в базу
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor() as cursor:
#             query = """
#             INSERT INTO faceNet (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path, emb_hash)
#             VALUES (%s, %s, %s, %s, %s, %s, %s)
#             """
#             await cursor.execute(query, (
#                 patient_id, hospital_id, branch_id, palata_id,
#                 image_filename, embedding_filename, emb_hash
#             ))
#             await conn.commit()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в БД: {str(e)}")
#     finally:
#         conn.close()

#     store_time = time.perf_counter() - start_time

#     return JSONResponse(content={
#         "status": True,
#         "message": "Эмбеддинг успешно сохранён.",
#         "store_time": store_time
#     })

# @app.post("/compare-face/")
# async def compare_face(
#     request: Request,
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
#     ):

#     await check_permission(user, request)

#     start_time = time.perf_counter()

#     # Читаем изображение
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

#     # Обнаружение лица
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

#     face_data = detection_results[0]
#     x, y, width, height = face_data['box']
#     face_img = img[y:y+height, x:x+width]

#     # Вычисляем эмбеддинг
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     query_embedding = embeddings[0]

#     # Вычисляем хэш эмбеддинга
#     query_hash = calculate_hash(query_embedding)

#     try:
#         conn = await get_db_connection()
#         async with conn.cursor(aiomysql.DictCursor) as cursor:
#             # Проверяем, есть ли уже такой эмбеддинг в базе
#             await cursor.execute("SELECT * FROM faceNet WHERE emb_hash = %s", (query_hash,))
#             duplicate = await cursor.fetchone()

#             if duplicate:
#                 return JSONResponse(content={
#                     "status": False,
#                     "message": "Такое же изображение уже использовалось.",
#                     "compare_time": time.perf_counter() - start_time
#                 })

#             # Получаем все эмбеддинги из базы
#             await cursor.execute("SELECT emb_path FROM faceNet")
#             rows = await cursor.fetchall()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при получении данных из базы: {str(e)}")
#     finally:
#         conn.close()

#     best_similarity = -1.0

#     for row in rows:
#         emb_path = row["emb_path"]
#         if not os.path.exists(emb_path):
#             continue
#         stored_embedding = np.load(emb_path)
#         similarity = cosine_similarity(query_embedding, stored_embedding)
#         if similarity > best_similarity:
#             best_similarity = similarity

#     compare_time = time.perf_counter() - start_time
#     match_percentage = float(best_similarity * 100)

#     if best_similarity >= SIMILARITY_THRESHOLD:
#         return JSONResponse(content={
#             "status": True,
#             "match_percentage": match_percentage,
#             "compare_time": compare_time
#         })
#     else:
#         return JSONResponse(content={
#             "status": False,
#             "message": "Совпадений не найдено.",
#             "match_percentage": match_percentage,
#             "compare_time": compare_time
#         })


# @app.post("/compare-face-qr/")
# async def compare_face_qr(
#     request: Request,
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
# ):
#     start_time = time.perf_counter()

#     # Читаем изображение
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

#     # Обнаружение лица
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

#     face_data = detection_results[0]
#     x, y, width, height = face_data['box']
#     face_img = img[y:y+height, x:x+width]

#     # Вычисляем эмбеддинг
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     query_embedding = embeddings[0]

#     # Вычисляем хэш эмбеддинга (можно использовать для логирования)
#     query_hash = calculate_hash(query_embedding)

#     try:
#         conn = await get_db_connection()
#         async with conn.cursor(aiomysql.DictCursor) as cursor:
#             # Получаем эмбеддинги и patient_id из таблицы faceNet (поля created_at там нет)
#             await cursor.execute("SELECT emb_path, patient_id FROM faceNet")
#             rows = await cursor.fetchall()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при получении данных из базы: {str(e)}")
#     finally:
#         conn.close()

#     best_similarity = -1.0
#     best_patient_id = None

#     for row in rows:
#         emb_path = row["emb_path"]
#         if not os.path.exists(emb_path):
#             continue
#         stored_embedding = np.load(emb_path)
#         similarity = cosine_similarity(query_embedding, stored_embedding)
#         if similarity > best_similarity:
#             best_similarity = similarity
#             best_patient_id = row["patient_id"]

#     # Если сходство превышает порог, считаем лицо распознанным
#     if best_similarity >= SIMILARITY_THRESHOLD:
#         # Записываем запись в таблицу QR, где created_at устанавливается автоматически
#         try:
#             conn = await get_db_connection()
#             async with conn.cursor() as cursor:
#                 insert_query = "INSERT INTO QR (status, patient_id) VALUES (%s, %s)"
#                 await cursor.execute(insert_query, (True, best_patient_id))
#                 await conn.commit()
#         except Exception as e:
#             print(f"Ошибка записи в таблицу QR: {e}")
#         finally:
#             conn.close()

#         return JSONResponse(content={
#             "status": True,
#             "patient_id": best_patient_id,
#             "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         })
#     else:
#         return JSONResponse(content={
#             "status": False,
#             "patient_id": None,
#             "created_at": None
#         })

# @app.get("/get-face-data/")
# async def get_face_data(
#     request: Request,
#     start_date: Optional[str] = Query(None, description="Начальная дата (YYYY-MM-DD)"),
#     end_date: Optional[str] = Query(None, description="Конечная дата (YYYY-MM-DD)"),
#     hospital_id: Optional[str] = Query(None, description="ID больницы"),
#     patient_id: Optional[str] = Query(None, description="ID пациента"),
#     user: dict = Depends(get_current_user),
# ):
#     await check_permission(user, request)

#     # ✅ Проверяем, что даты в правильном формате (если переданы)
#     def validate_date(date_str):
#         try:
#             return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
#         except ValueError:
#             raise HTTPException(status_code=400, detail=f"Неверный формат даты: {date_str}. Используйте YYYY-MM-DD")

#     if start_date:
#         start_date = validate_date(start_date)
#     if end_date:
#         end_date = validate_date(end_date)

#     try:
#         with mysql.connector.connect(**DB_CONFIG) as conn:
#             with conn.cursor(dictionary=True) as cursor:
#                 query = """
#                 SELECT id, patient_id, hospital_id, status, similarity_percentage, comparison_time, timestamp 
#                 FROM faceData
#                 """
#                 params = []

#                 # ✅ Фильтрация по дате, hospital_id и patient_id
#                 conditions = []
#                 if start_date and end_date:
#                     conditions.append("timestamp BETWEEN %s AND %s")
#                     params.extend([start_date + " 00:00:00", end_date + " 23:59:59"])
#                 elif start_date:
#                     conditions.append("timestamp >= %s")
#                     params.append(start_date + " 00:00:00")
#                 elif end_date:
#                     conditions.append("timestamp <= %s")
#                     params.append(end_date + " 23:59:59")

#                 if hospital_id:
#                     conditions.append("hospital_id = %s")
#                     params.append(hospital_id)

#                 if patient_id:
#                     conditions.append("patient_id = %s")
#                     params.append(patient_id)

#                 # ✅ Добавляем WHERE только если есть условия
#                 if conditions:
#                     query += " WHERE " + " AND ".join(conditions)

#                 query += " ORDER BY timestamp DESC"

#                 cursor.execute(query, params)
#                 results = cursor.fetchall()

#         return {"count": len(results), "data": results}

#     except mysql.connector.Error as e:
#         return {"error": f"Ошибка при получении данных: {e}"}




# @app.post("/compare-face/")
# async def compare_face(
#     request: Request,
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
# ):
#     """Сравнивает лицо со всеми сохранёнными эмбеддингами в faceNet."""
#     await check_permission(user, request)
#     start_time = time.perf_counter()

#     # 1. Считываем изображение, получаем лицо и эмбеддинг
#     img = load_image_from_upload(file)
#     face_img = detect_face(img)
#     query_embedding = get_embedding(face_img)

#     # 2. Проверяем, не существует ли уже точного дубликата
#     query_hash = calculate_hash(query_embedding)
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor(aiomysql.DictCursor) as cursor:
#             await cursor.execute("SELECT id FROM faceNet WHERE emb_hash = %s", (query_hash,))
#             exact_duplicate = await cursor.fetchone()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при работе с БД: {str(e)}")
#     finally:
#         conn.close()

#     if exact_duplicate:
#         # Если нашли точный дубликат — возвращаем сообщение
#         return JSONResponse(content={
#             "status": False,
#             "message": "Такое же изображение уже использовалось.",
#             "compare_time": time.perf_counter() - start_time
#         })

#     # 3. Получаем все эмбеддинги из базы
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor(aiomysql.DictCursor) as cursor:
#             await cursor.execute("SELECT emb_path FROM faceNet")
#             rows = await cursor.fetchall()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при получении данных из БД: {str(e)}")
#     finally:
#         conn.close()

#     # 4. Перебираем все эмбеддинги и ищем максимально похожий
#     best_similarity = -1.0
#     for row in rows:
#         emb_path = row["emb_path"]
#         stored_embedding = load_embedding_into_cache(emb_path)
#         if stored_embedding is None:
#             continue
#         sim = cosine_similarity(query_embedding, stored_embedding)
#         if sim > best_similarity:
#             best_similarity = sim

#     compare_time = time.perf_counter() - start_time
#     match_percentage = best_similarity * 100.0

#     if best_similarity >= SIMILARITY_THRESHOLD:
#         return JSONResponse(content={
#             "status": True,
#             "match_percentage": match_percentage,
#             "compare_time": compare_time
#         })
#     else:
#         return JSONResponse(content={
#             "status": False,
#             "message": "Совпадений не найдено.",
#             "match_percentage": match_percentage,
#             "compare_time": compare_time
#         })


# @app.post("/process-patient")
# async def process_patient(
#     request: Request,
#     patient_id: str = Form(...),
#     hospital_id: str = Form(...),
#     branch_id: str = Form(...),
#     palata_id: str = Form(...),
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
# ):
#     """Принимает изображение, обнаруживает лицо, вычисляет эмбеддинг и сохраняет его в БД."""
#     await check_permission(user, request)
#     start_time = time.perf_counter()

#     # 1. Считываем изображение и получаем лицо
#     img = load_image_from_upload(file)
#     face_img = detect_face(img)

#     # 2. Получаем эмбеддинг и его хэш
#     embedding = get_embedding(face_img)
#     emb_hash = calculate_hash(embedding)

#     # 3. Проверяем дубликат в БД
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor() as cursor:
#             await cursor.execute("SELECT id FROM faceNet WHERE emb_hash = %s", (emb_hash,))
#             duplicate = await cursor.fetchone()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при запросе к БД: {str(e)}")
#     finally:
#         conn.close()

#     if duplicate:
#         return JSONResponse(content={
#             "status": False,
#             "message": "Это лицо уже сохранено."
#         })

#     # 4. Сохраняем лицо и эмбеддинг на диск  
#     image_filename = f"images/{uuid.uuid4()}.jpg"
#     embedding_filename = f"embeddings/{uuid.uuid4()}.npy"
#     cv2.imwrite(image_filename, face_img)
#     np.save(embedding_filename, embedding)

#     # 5. Добавляем запись в базу
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor() as cursor:
#             query = """
#             INSERT INTO faceNet (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path, emb_hash)
#             VALUES (%s, %s, %s, %s, %s, %s, %s)
#             """
#             await cursor.execute(query, (
#                 patient_id, hospital_id, branch_id, palata_id,
#                 image_filename, embedding_filename, emb_hash
#             ))
#             await conn.commit()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в БД: {str(e)}")
#     finally:
#         conn.close()

#     elapsed_time = time.perf_counter() - start_time
#     return JSONResponse(content={
#         "status": True,
#         "message": "Эмбеддинг успешно сохранён.",
#         "store_time": elapsed_time
#     })


# import os
# import uuid
# import hashlib
# import time
# import json
# from datetime import datetime
# from typing import Optional
# import aiomysql
# import mysql.connector
# import numpy as np
# import cv2
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, Request
# from fastapi.responses import JSONResponse

# # MTCNN и FaceNet
# from mtcnn import MTCNN
# from keras_facenet import FaceNet 

# # Ваши роутеры авторизации/регистрации
# from auth import router as auth_router, get_current_user
# from register import router as register_router

# # -------------------------------
# # 1. Настройки и глобальные объекты
# # -------------------------------

# app = FastAPI()

# app.include_router(auth_router)
# app.include_router(register_router)

# DB_CONFIG = {
#     "host": "face.tabet-kitap.kz",
#     "user": "fastapi_user",
#     "password": "secure_password",
#     "database": "face_db",
#     "port": 3306
# }

# # Директории хранения
# os.makedirs("images", exist_ok=True)
# os.makedirs("embeddings", exist_ok=True)

# # Инициализация детектора лиц и модели эмбеддингов
# detector = MTCNN()
# embedder = FaceNet()

# # Порог схожести
# SIMILARITY_THRESHOLD = 0.7

# # -------------------------------
# # 2. Функции работы с БД
# # -------------------------------

# async def get_db_connection():
#     """Асинхронное получение подключения к базе данных aiomysql."""
#     conn = await aiomysql.connect(
#         host=DB_CONFIG["host"],
#         port=DB_CONFIG["port"],
#         user=DB_CONFIG["user"],
#         password=DB_CONFIG["password"],
#         db=DB_CONFIG["database"],
#         autocommit=True
#     )
#     return conn

# def get_sync_connection():
#     """Синхронное подключение (для простых get-запросов через mysql.connector)."""
#     return mysql.connector.connect(
#         host=DB_CONFIG["host"],
#         user=DB_CONFIG["user"],
#         password=DB_CONFIG["password"],
#         database=DB_CONFIG["database"],
#         port=DB_CONFIG["port"]
#     )

# # -------------------------------
# # 3. Функции распознавания лица
# # -------------------------------

# def load_image_from_upload(file: UploadFile) -> np.ndarray:
#     """Считывает UploadFile в OpenCV-изображение (BGR)."""
#     contents = file.file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")
#     return img

# def detect_face(img: np.ndarray) -> np.ndarray:
#     """Обнаруживает первое лицо, вырезает и возвращает его как np.ndarray."""
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")
    
#     # Для упрощения берём первое найденное лицо
#     face_data = detection_results[0]
#     x, y, w, h = face_data['box']
    
#     # Корректируем границы, если они выходят за пределы изображения
#     x = max(x, 0)
#     y = max(y, 0)
#     w = max(w, 0)
#     h = max(h, 0)
    
#     face_img = img[y:y+h, x:x+w]
#     if face_img.size == 0:
#         raise HTTPException(status_code=400, detail="Невозможно вырезать лицо из изображения.")
#     return face_img

# def get_embedding(face_img: np.ndarray) -> np.ndarray:
#     """Возвращает эмбеддинг (512-мерный вектор) для вырезанного лица."""
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     return embeddings[0]

# def calculate_hash(embedding: np.ndarray) -> str:
#     """Вычисляет SHA256-хэш от эмбеддинга."""
#     return hashlib.sha256(embedding.tobytes()).hexdigest()

# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     """Вычисление косинусной схожести между двумя векторами."""
#     a_norm = np.linalg.norm(a)
#     b_norm = np.linalg.norm(b)
#     if a_norm == 0 or b_norm == 0:
#         return 0.0
#     return float(np.dot(a, b) / (a_norm * b_norm))

# # -------------------------------
# # 3.1 Кэширование эмбеддингов (опционально)
# # -------------------------------
# # Для примера используем простой словарь, где ключ — emb_path, значение — сам вектор.
# # В реальном проекте кэш или может быть больше «умным» (например, библиотека LRUCache и т.п.)

# EMBEDDINGS_CACHE = {}

# def load_embedding_into_cache(emb_path: str) -> np.ndarray:
#     """Загружает эмбеддинг из файла в кэш (если там ещё нет)."""
#     if emb_path not in EMBEDDINGS_CACHE:
#         if os.path.exists(emb_path):
#             EMBEDDINGS_CACHE[emb_path] = np.load(emb_path)
#         else:
#             # Логируем предупреждение, можно пропустить
#             EMBEDDINGS_CACHE[emb_path] = None
#     return EMBEDDINGS_CACHE[emb_path]

# # -------------------------------
# # 4. Проверка прав доступа
# # -------------------------------

# async def check_permission(user: dict, request: Request):
#     """Проверяет доступ пользователя к API."""
#     allowed_api_str = user.get("allowed_api", "[]")  # Получаем строку
#     try:
#         allowed_api = json.loads(allowed_api_str)  # Конвертируем JSON в список
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Ошибка в формате данных API")

#     # Текущее имя endpoint
#     api_name = request.scope["path"].lstrip("/")

#     # Если глобальный доступ (*) или конкретный роут в списке
#     if "*" in allowed_api or api_name in allowed_api:
#         return

#     raise HTTPException(
#         status_code=403,
#         detail=f"У вас нет доступа к API {api_name}"
#     )

# # -------------------------------
# # 5. Роутеры FastAPI
# # -------------------------------



# @app.post("/process-patient")
# async def process_patient(
#     request: Request,
#     patient_id: str = Form(...),
#     hospital_id: str = Form(...),
#     branch_id: str = Form(...),
#     palata_id: str = Form(...),
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
# ):
#     """Принимает изображение, обнаруживает лицо, вычисляет эмбеддинг и сохраняет его в БД."""
#     await check_permission(user, request)
#     start_time = time.perf_counter()

#     # 1. Считываем изображение и получаем лицо
#     img = load_image_from_upload(file)
#     face_img = detect_face(img)

#     # 2. Получаем эмбеддинг и его хэш
#     embedding = get_embedding(face_img)
#     emb_hash = calculate_hash(embedding)

#     # 3. Проверяем дубликат в БД
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor() as cursor:
#             await cursor.execute("SELECT id FROM faceNet WHERE emb_hash = %s", (emb_hash,))
#             duplicate = await cursor.fetchone()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при запросе к БД: {str(e)}")
#     finally:
#         conn.close()

#     if duplicate:
#         return JSONResponse(content={
#             "status": False,
#             "message": "Это лицо уже сохранено."
#         })

#     # 4. Сохраняем лицо и эмбеддинг на диск  
#     image_filename = f"images/{uuid.uuid4()}.jpg"
#     embedding_filename = f"embeddings/{uuid.uuid4()}.npy"
#     cv2.imwrite(image_filename, face_img)
#     np.save(embedding_filename, embedding)

#     # 5. Добавляем запись в базу
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor() as cursor:
#             # Вставляем новую запись
#             insert_query = """
#                 INSERT INTO faceNet (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path, emb_hash)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#             """
#             await cursor.execute(insert_query, (
#                 patient_id, hospital_id, branch_id, palata_id,
#                 image_filename, embedding_filename, emb_hash
#             ))

#             # Получаем ID, сгенерированный автоинкрементом
#             new_id = cursor.lastrowid

#             # Подтверждаем INSERT
#             await conn.commit()

#             # Теперь получаем created_at для этой записи
#             select_query = "SELECT created_at FROM faceNet WHERE id = %s"
#             await cursor.execute(select_query, (new_id,))
#             row = await cursor.fetchone()
#             if row:
#                 created_at_value = row[0]  # это будет объект datetime
#             else:
#                 created_at_value = None

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в БД: {str(e)}")
#     finally:
#         conn.close()

#     elapsed_time = time.perf_counter() - start_time

#     # Возвращаем created_at в JSON (преобразуем datetime в строку isoformat)
#     return JSONResponse(content={
#         "status": True,
#         "message": "Эмбеддинг успешно сохранён.",
#         "store_time": elapsed_time,
#         "created_at": created_at_value.isoformat() if created_at_value else None
#     })

# @app.post("/compare-face/")
# async def compare_face(
#     request: Request,
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
# ):
#     """Сравнивает лицо со всеми сохранёнными эмбеддингами в faceNet."""
#     await check_permission(user, request)

#     start_time = time.perf_counter()

#     # 1. Считываем изображение и получаем эмбеддинг
#     img = load_image_from_upload(file)
#     face_img = detect_face(img)
#     query_embedding = get_embedding(face_img)

#     # 2. Получаем все эмбеддинги из базы (ВАЖНО: выбираем поля, которые нужны!)
#     # Предположим, что в таблице faceNet имеются столбцы: id, emb_path, patient_id, hospital_id
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor(aiomysql.DictCursor) as cursor:
#             await cursor.execute("""
#                 SELECT emb_path, patient_id, hospital_id 
#                 FROM faceNet
#             """)
#             rows = await cursor.fetchall()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при получении данных из БД: {str(e)}")
#     finally:
#         conn.close()

#     # 3. Ищем максимально похожий эмбеддинг
#     best_similarity = -1.0
#     matched_patient_id = None
#     matched_hospital_id = None

#     for row in rows:
#         emb_path = row["emb_path"]
#         stored_embedding = load_embedding_into_cache(emb_path)
#         if stored_embedding is None:
#             continue

#         sim = cosine_similarity(query_embedding, stored_embedding)
#         if sim > best_similarity:
#             best_similarity = sim
#             matched_patient_id = row["patient_id"]
#             matched_hospital_id = row["hospital_id"]

#     # Высчитываем процент сходства и время сравнения
#     match_percentage = best_similarity * 100.0
#     compare_time = time.perf_counter() - start_time

#     # Определяем статус (True/False) по порогу сходства
#     status = best_similarity >= SIMILARITY_THRESHOLD

#     # 4. Всегда записываем результат в faceData
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor() as cursor:
#             await cursor.execute(
#                 """
#                 INSERT INTO faceData
#                     (patient_id, hospital_id, status, similarity_percentage, comparison_time, timestamp)
#                 VALUES
#                     (%s, %s, %s, %s, %s, NOW())
#                 """,
#                 (
#                     matched_patient_id,    # Могут быть None, если не нашли
#                     matched_hospital_id,   # Могут быть None, если не нашли
#                     status,                # True или False
#                     match_percentage,      # % сходства
#                     compare_time           # время сравнения
#                 )
#             )
#             await conn.commit()
#     except Exception as e:
#         print(f"❌ Ошибка при записи в БД: {e}")
#     finally:
#         conn.close()

#     # 5. Возвращаем результат
#     if status:
#         return JSONResponse(content={
#             "status": True,
#             "match_percentage": match_percentage,
#             "compare_time": compare_time
#         })
#     else:
#         return JSONResponse(content={
#             "status": False,
#             "message": "Совпадений не найдено.",
#             "match_percentage": match_percentage,
#             "compare_time": compare_time
#         })


# @app.post("/compare-face-qr/")
# async def compare_face_qr(
#     request: Request,
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
# ):
#     """
#     Сравнивает лицо со всеми эмбеддингами, если есть совпадение выше порога —
#     записывает результат в таблицу QR со статусом True.
#     """
#     await check_permission(user, request)
#     start_time = time.perf_counter()

#     # 1. Считываем изображение, получаем лицо, эмбеддинг
#     img = load_image_from_upload(file)
#     face_img = detect_face(img)
#     query_embedding = get_embedding(face_img)
#     query_hash = calculate_hash(query_embedding)  # для логирования/проверки

#     # 2. Получаем все emb_path + patient_id
#     try:
#         conn = await get_db_connection()
#         async with conn.cursor(aiomysql.DictCursor) as cursor:
#             await cursor.execute("SELECT emb_path, patient_id FROM faceNet")
#             rows = await cursor.fetchall()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при получении данных из БД: {str(e)}")
#     finally:
#         conn.close()

#     # 3. Ищем лучшее совпадение
#     best_similarity = -1.0
#     best_patient_id = None
#     for row in rows:
#         emb_path = row["emb_path"]
#         stored_embedding = load_embedding_into_cache(emb_path)
#         if stored_embedding is None:
#             continue

#         sim = cosine_similarity(query_embedding, stored_embedding)
#         if sim > best_similarity:
#             best_similarity = sim
#             best_patient_id = row["patient_id"]

#     # 4. Если сходство выше порога — записываем в таблицу QR
#     matched = best_similarity >= SIMILARITY_THRESHOLD
#     if matched and best_patient_id is not None:
#         try:
#             conn = await get_db_connection()
#             async with conn.cursor() as cursor:
#                 insert_query = "INSERT INTO QR (status, patient_id) VALUES (%s, %s)"
#                 await cursor.execute(insert_query, (True, best_patient_id))
#                 await conn.commit()
#         except Exception as e:
#             print(f"Ошибка записи в таблицу QR: {e}")
#         finally:
#             conn.close()

#         return JSONResponse(content={
#             "status": True,
#             "patient_id": best_patient_id,
#             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         })
#     else:
#         return JSONResponse(content={
#             "status": False,
#             "patient_id": None,
#             "created_at": None
#         })


# @app.get("/get-face-data/")
# async def get_face_data(
#     request: Request,
#     start_date: Optional[str] = Query(None, description="Начальная дата (YYYY-MM-DD)"),
#     end_date: Optional[str] = Query(None, description="Конечная дата (YYYY-MM-DD)"),
#     hospital_id: Optional[str] = Query(None, description="ID больницы"),
#     patient_id: Optional[str] = Query(None, description="ID пациента"),
#     user: dict = Depends(get_current_user),
# ):
#     """Возвращает список данных по сравнению лиц с возможностью фильтрации по дате, больнице, пациенту."""
#     await check_permission(user, request)

#     def validate_date(date_str: str) -> str:
#         """Проверка формата даты YYYY-MM-DD."""
#         try:
#             dt = datetime.strptime(date_str, "%Y-%m-%d")
#             return dt.strftime("%Y-%m-%d")
#         except ValueError:
#             raise HTTPException(status_code=400, detail=f"Неверный формат даты: {date_str}. Используйте YYYY-MM-DD")

#     if start_date:
#         start_date = validate_date(start_date)
#     if end_date:
#         end_date = validate_date(end_date)

#     try:
#         with get_sync_connection() as conn:
#             with conn.cursor(dictionary=True) as cursor:
#                 query = """
#                 SELECT 
#                     id, 
#                     patient_id, 
#                     hospital_id, 
#                     status, 
#                     similarity_percentage, 
#                     comparison_time, 
#                     timestamp 
#                 FROM faceData
#                 """
#                 params = []
#                 conditions = []

#                 if start_date and end_date:
#                     conditions.append("timestamp BETWEEN %s AND %s")
#                     params.extend([start_date + " 00:00:00", end_date + " 23:59:59"])
#                 elif start_date:
#                     conditions.append("timestamp >= %s")
#                     params.append(start_date + " 00:00:00")
#                 elif end_date:
#                     conditions.append("timestamp <= %s")
#                     params.append(end_date + " 23:59:59")

#                 if hospital_id:
#                     conditions.append("hospital_id = %s")
#                     params.append(hospital_id)

#                 if patient_id:
#                     conditions.append("patient_id = %s")
#                     params.append(patient_id)

#                 if conditions:
#                     query += " WHERE " + " AND ".join(conditions)

#                 query += " ORDER BY timestamp DESC"
#                 cursor.execute(query, params)
#                 results = cursor.fetchall()

#         return {"count": len(results), "data": results}

#     except mysql.connector.Error as e:
#         return {"error": f"Ошибка при получении данных: {e}"}


import os
import uuid
import hashlib
import time
import json
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import aiomysql
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse

# MTCNN и FaceNet
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Ваши роутеры авторизации/регистрации (если есть)
from auth import router as auth_router, get_current_user
from register import router as register_router

app = FastAPI()

app.include_router(auth_router)
app.include_router(register_router)

DB_CONFIG = {
    "host": "face.tabet-kitap.kz",
    "user": "fastapi_user",
    "password": "secure_password",
    "database": "face_db",
    "port": 3306
}

# --------------------
# Пул соединений aiomysql
# --------------------
aiomysql_pool = None

@app.on_event("startup")
async def on_startup():
    global aiomysql_pool
    # Инициализируем пул соединений
    aiomysql_pool = await aiomysql.create_pool(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        db=DB_CONFIG["database"],
        autocommit=True,
        minsize=1,
        maxsize=10
    )
    # Загружаем эмбеддинги в память
    await load_embeddings_on_startup()

@app.on_event("shutdown")
async def on_shutdown():
    global aiomysql_pool
    if aiomysql_pool:
        aiomysql_pool.close()
        await aiomysql_pool.wait_closed()

async def get_db_connection():
    """Берёт соединение из пула aiomysql."""
    global aiomysql_pool
    if aiomysql_pool is None:
        raise HTTPException(status_code=500, detail="Пул соединений не инициализирован")
    return await aiomysql_pool.acquire()

async def release_db_connection(conn):
    """Возвращает соединение в пул."""
    global aiomysql_pool
    if aiomysql_pool and conn:
        aiomysql_pool.release(conn)

# --------------------------------------------------
# Глобальный список для хранения эмбеддингов в памяти:
# (patient_id, hospital_id, embedding (np.ndarray), emb_path)
# --------------------------------------------------
EMBEDDINGS_IN_MEMORY: List[Tuple[str, str, np.ndarray, str]] = []

async def load_embeddings_on_startup():
    """
    Загружает все эмбеддинги из таблицы faceNet
    и сохраняет их в EMBEDDINGS_IN_MEMORY (в оперативной памяти).
    Вызывается один раз при старте приложения.
    """
    global EMBEDDINGS_IN_MEMORY
    conn = None
    try:
        conn = await get_db_connection()
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("SELECT patient_id, hospital_id, emb_path FROM faceNet")
            rows = await cursor.fetchall()

            temp_list = []
            for row in rows:
                emb_path = row["emb_path"]
                if os.path.exists(emb_path):
                    embedding = np.load(emb_path)
                    temp_list.append(
                        (row["patient_id"], row["hospital_id"], embedding, emb_path)
                    )

            EMBEDDINGS_IN_MEMORY = temp_list
            print(f"[startup] Загружено {len(EMBEDDINGS_IN_MEMORY)} эмбеддингов в память.")
    except Exception as e:
        print(f"Ошибка при загрузке эмбеддингов: {e}")
    finally:
        if conn:
            await release_db_connection(conn)

# Директории для файлов
os.makedirs("images", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

detector = MTCNN()
embedder = FaceNet()
SIMILARITY_THRESHOLD = 0.7

# Кэш для хранения загруженных с диска эмбеддингов (если вам это ещё потребуется)
EMBEDDINGS_CACHE = {}

def load_embedding_into_cache(emb_path: str) -> np.ndarray:
    """Загружает эмбеддинг из файла в кэш (если там ещё нет)."""
    if emb_path not in EMBEDDINGS_CACHE:
        if os.path.exists(emb_path):
            EMBEDDINGS_CACHE[emb_path] = np.load(emb_path)
        else:
            EMBEDDINGS_CACHE[emb_path] = None
    return EMBEDDINGS_CACHE[emb_path]

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")
    return img

def load_image_from_base64(b64_string: str) -> np.ndarray:
    """
    Декодирует Base64-строку в np.ndarray (BGR-изображение).
    Убирает, при необходимости, префикс 'data:image/...base64,'.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        image_data = base64.b64decode(b64_string)
    except Exception as e:
        raise ValueError(f"Ошибка при декодировании base64: {e}")

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Невозможно прочитать изображение из base64.")
    return img

def detect_face(img: np.ndarray) -> np.ndarray:
    detection_results = detector.detect_faces(img)
    if not detection_results:
        raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

    face_data = detection_results[0]
    x, y, w, h = face_data['box']
    x = max(x, 0)
    y = max(y, 0)
    face_img = img[y:y+h, x:x+w]
    if face_img.size == 0:
        raise HTTPException(status_code=400, detail="Невозможно вырезать лицо.")
    return face_img

def get_embedding(face_img: np.ndarray) -> np.ndarray:
    embeddings = embedder.embeddings([face_img])
    if embeddings.size == 0:
        raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
    return embeddings[0]

def calculate_hash(embedding: np.ndarray) -> str:
    return hashlib.sha256(embedding.tobytes()).hexdigest()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

# Проверка прав
async def check_permission(user: dict, request: Request):
    allowed_api_str = user.get("allowed_api", "[]")
    try:
        allowed_api = json.loads(allowed_api_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Ошибка в формате allowed_api")

    api_name = request.scope["path"].lstrip("/")
    if "*" in allowed_api or api_name in allowed_api:
        return
    raise HTTPException(status_code=403, detail=f"Нет доступа к API {api_name}")

@app.post("/process-patient")
async def process_patient(
    request: Request,
    patient_id: str = Form(...),
    hospital_id: str = Form(...),
    branch_id: str = Form(...),
    palata_id: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Принимает файл (UploadFile), обнаруживает лицо, вычисляет эмбеддинг и сохраняет его в БД.
    Также добавляет новый эмбеддинг в EMBEDDINGS_IN_MEMORY.
    """
    await check_permission(user, request)
    start_time = time.perf_counter()

    # Считываем изображение => детект => эмбеддинг
    img = load_image_from_upload(file)
    face_img = detect_face(img)
    embedding = get_embedding(face_img)
    emb_hash = calculate_hash(embedding)

    # Проверка дубликата
    conn = None
    try:
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT id FROM faceNet WHERE emb_hash = %s", (emb_hash,))
            duplicate = await cursor.fetchone()
        if duplicate:
            return JSONResponse(content={
                "status": False,
                "message": "Это лицо уже сохранено."
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при запросе к БД: {str(e)}")
    finally:
        if conn:
            await release_db_connection(conn)

    # Сохраняем файлы
    image_filename = f"images/{uuid.uuid4()}.jpg"
    embedding_filename = f"embeddings/{uuid.uuid4()}.npy"
    cv2.imwrite(image_filename, face_img)
    np.save(embedding_filename, embedding)

    # Записываем в БД
    conn = None
    created_at_value = None
    try:
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            insert_query = """
                INSERT INTO faceNet 
                (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path, emb_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            await cursor.execute(insert_query, (
                patient_id, hospital_id, branch_id, palata_id,
                image_filename, embedding_filename, emb_hash
            ))
            new_id = cursor.lastrowid

            # Получаем created_at
            select_query = "SELECT created_at FROM faceNet WHERE id = %s"
            await cursor.execute(select_query, (new_id,))
            row = await cursor.fetchone()
            if row:
                created_at_value = row[0]
            await conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в БД: {str(e)}")
    finally:
        if conn:
            await release_db_connection(conn)

    # Добавляем новую запись в EMBEDDINGS_IN_MEMORY
    EMBEDDINGS_IN_MEMORY.append((patient_id, hospital_id, embedding, embedding_filename))

    elapsed_time = time.perf_counter() - start_time
    return JSONResponse(content={
        "status": True,
        "message": "Эмбеддинг успешно сохранён.",
        "store_time": elapsed_time,
        "created_at": created_at_value.isoformat() if created_at_value else None
    })


# -----------------------------------------------------------------------------
# process-patient-base64 (через Base64-строку)
# -----------------------------------------------------------------------------
@app.post("/process-patient-base64/")
async def process_patient_base64(
    request: Request,
    patient_id: str = Form(...),
    hospital_id: str = Form(...),
    branch_id: str = Form(...),
    palata_id: str = Form(...),
    file_base64: str = Form(...),  # Base64-строка
    user: dict = Depends(get_current_user),
):
    """
    Принимает изображение в виде Base64-строки,
    обнаруживает лицо, вычисляет эмбеддинг и сохраняет его в БД.
    Аналог process-patient, но с Base64 вместо UploadFile.
    """
    await check_permission(user, request)
    start_time = time.perf_counter()

    # 1. Декодируем base64-строку в np.ndarray
    try:
        img = load_image_from_base64(file_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при декодировании base64: {str(e)}")

    # 2. Детектируем лицо
    face_img = detect_face(img)

    # 3. Получаем эмбеддинг и хэш
    embedding = get_embedding(face_img)
    emb_hash = calculate_hash(embedding)

    # 4. Проверяем, не дубликат ли
    conn = None
    try:
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT id FROM faceNet WHERE emb_hash = %s", (emb_hash,))
            duplicate = await cursor.fetchone()
        if duplicate:
            return JSONResponse(content={
                "status": False,
                "message": "Это лицо уже сохранено."
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при запросе к БД: {str(e)}")
    finally:
        if conn:
            await release_db_connection(conn)

    # 5. Сохраняем файлы (изображение и эмбеддинг) на диск
    image_filename = f"images/{uuid.uuid4()}.jpg"
    embedding_filename = f"embeddings/{uuid.uuid4()}.npy"
    cv2.imwrite(image_filename, face_img)
    np.save(embedding_filename, embedding)

    # 6. Записываем в БД
    conn = None
    created_at_value = None
    try:
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            insert_query = """
                INSERT INTO faceNet 
                (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path, emb_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            await cursor.execute(insert_query, (
                patient_id, hospital_id, branch_id, palata_id,
                image_filename, embedding_filename, emb_hash
            ))
            new_id = cursor.lastrowid

            # Получаем created_at
            select_query = "SELECT created_at FROM faceNet WHERE id = %s"
            await cursor.execute(select_query, (new_id,))
            row = await cursor.fetchone()
            if row:
                created_at_value = row[0]
            await conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в БД: {str(e)}")
    finally:
        if conn:
            await release_db_connection(conn)

    # 7. Добавляем новую запись в EMBEDDINGS_IN_MEMORY
    EMBEDDINGS_IN_MEMORY.append((patient_id, hospital_id, embedding, embedding_filename))

    elapsed_time = time.perf_counter() - start_time
    return JSONResponse(content={
        "status": True,
        "message": "Эмбеддинг успешно сохранён (Base64).",
        "store_time": elapsed_time,
        "created_at": created_at_value.isoformat() if created_at_value else None
    })


@app.post("/compare-face/")
async def compare_face(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Сравнивает загруженное лицо со всеми эмбеддингами в памяти (EMBEDDINGS_IN_MEMORY).
    Записывает результат в таблицу faceData.
    """
    await check_permission(user, request)
    start_time = time.perf_counter()

    # Получаем эмбеддинг загруженного лица
    img = load_image_from_upload(file)
    face_img = detect_face(img)
    query_embedding = get_embedding(face_img)

    # Ищем самый похожий эмбеддинг (линейный перебор по EMBEDDINGS_IN_MEMORY)
    best_similarity = -1.0
    matched_patient_id = None
    matched_hospital_id = None

    for (p_id, h_id, stored_embedding, emb_path) in EMBEDDINGS_IN_MEMORY:
        if stored_embedding is None:
            continue
        sim = cosine_similarity(query_embedding, stored_embedding)
        if sim > best_similarity:
            best_similarity = sim
            matched_patient_id = p_id
            matched_hospital_id = h_id

    match_percentage = best_similarity * 100.0
    compare_time = time.perf_counter() - start_time
    status = best_similarity >= SIMILARITY_THRESHOLD

    # Пишем лог в faceData
    conn = None
    try:
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO faceData
                    (patient_id, hospital_id, status, similarity_percentage, comparison_time, timestamp)
                VALUES
                    (%s, %s, %s, %s, %s, NOW())
                """,
                (
                    matched_patient_id,
                    matched_hospital_id,
                    status,
                    match_percentage,
                    compare_time
                )
            )
            await conn.commit()
    except Exception as e:
        print(f"Ошибка при записи в faceData: {e}")
    finally:
        if conn:
            await release_db_connection(conn)

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
    user: dict = Depends(get_current_user),
):
    """
    Сравнение лица и, при успехе, запись результата в таблицу QR.
    """
    await check_permission(user, request)
    start_time = time.perf_counter()

    # Получаем эмбеддинг загруженного лица
    img = load_image_from_upload(file)
    face_img = detect_face(img)
    query_embedding = get_embedding(face_img)

    best_similarity = -1.0
    best_patient_id = None

    for (p_id, h_id, stored_embedding, emb_path) in EMBEDDINGS_IN_MEMORY:
        if stored_embedding is None:
            continue
        sim = cosine_similarity(query_embedding, stored_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_patient_id = p_id

    matched = best_similarity >= SIMILARITY_THRESHOLD
    if matched and best_patient_id is not None:
        # Запись в таблицу QR
        conn = None
        try:
            conn = await get_db_connection()
            async with conn.cursor() as cursor:
                insert_query = "INSERT INTO QR (status, patient_id) VALUES (%s, %s)"
                await cursor.execute(insert_query, (True, best_patient_id))
                await conn.commit()
        except Exception as e:
            print(f"Ошибка записи в таблицу QR: {e}")
        finally:
            if conn:
                await release_db_connection(conn)

        return JSONResponse(content={
            "status": True,
            "patient_id": best_patient_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        return JSONResponse(content={
            "status": False,
            "patient_id": None,
            "created_at": None
        })

@app.get("/get-face-data/")
async def get_face_data(
    request: Request,
    start_date: Optional[str] = Query(None, description="Начальная дата (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Конечная дата (YYYY-MM-DD)"),
    hospital_id: Optional[str] = Query(None, description="ID больницы"),
    patient_id: Optional[str] = Query(None, description="ID пациента"),
    user: dict = Depends(get_current_user),
):
    """
    Возвращает записи из faceData с фильтрами по датам, patient_id, hospital_id.
    """
    await check_permission(user, request)

    def validate_date(date_str: str) -> str:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Неверный формат даты: {date_str}. Используйте YYYY-MM-DD"
            )

    if start_date:
        start_date = validate_date(start_date)
    if end_date:
        end_date = validate_date(end_date)

    query = """
        SELECT
            id,
            patient_id,
            hospital_id,
            status,
            similarity_percentage,
            comparison_time,
            timestamp
        FROM faceData
    """
    conditions = []
    params = []

    if start_date and end_date:
        conditions.append("timestamp BETWEEN %s AND %s")
        params.extend([start_date + " 00:00:00", end_date + " 23:59:59"])
    elif start_date:
        conditions.append("timestamp >= %s")
        params.append(start_date + " 00:00:00")
    elif end_date:
        conditions.append("timestamp <= %s")
        params.append(end_date + " 23:59:59")

    if hospital_id:
        conditions.append("hospital_id = %s")
        params.append(hospital_id)
    if patient_id:
        conditions.append("patient_id = %s")
        params.append(patient_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC"

    conn = None
    results: List[Dict[str, Any]] = []
    try:
        conn = await get_db_connection()
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, params)
            results = await cursor.fetchall()
    except Exception as e:
        return {"error": f"Ошибка при получении данных: {e}"}
    finally:
        if conn:
            await release_db_connection(conn)

    return {"count": len(results), "data": results}
