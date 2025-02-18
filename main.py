# import time
# import cv2
# import numpy as np
# from mtcnn import MTCNN
# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from fastapi.responses import JSONResponse
# from keras_facenet import FaceNet
# import aiomysql

# app = FastAPI()

# # Инициализируем детектор лиц (MTCNN)
# detector = MTCNN()

# # Инициализируем объект FaceNet, модель будет загружена автоматически при первом вызове
# embedder = FaceNet()

# # Конфигурация базы данных
# DB_CONFIG = {
#     "host": "face.tabet-kitap.kz",
#     "user": "fastapi_user",
#     "password": "secure_password",
#     "database": "face_db",
#     "port": 3306
# }

# async def get_db_connection():
#     return await aiomysql.connect(
#         host=DB_CONFIG["host"],
#         user=DB_CONFIG["user"],
#         password=DB_CONFIG["password"],
#         db=DB_CONFIG["database"],
#         port=DB_CONFIG["port"]
#     )

# def cosine_similarity(a, b):
#     """Вычисление косинусной схожести между двумя векторами."""
#     a_norm = np.linalg.norm(a)
#     b_norm = np.linalg.norm(b)
#     if a_norm == 0 or b_norm == 0:
#         return 0.0
#     similarity = np.dot(a, b) / (a_norm * b_norm)
#     return similarity

# # Порог схожести для определения совпадения (значение подбирается экспериментально)
# SIMILARITY_THRESHOLD = 0.7

# @app.post("/process-patient/")
# async def process_patient(
#     patient_id: str = Form(...),
#     hospital_id: str = Form(...),
#     branch_id: str = Form(...),
#     palata_id: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     """
#     API для сохранения эмбеддинга лица и данных пациента в базу данных.
#     """
#     start_time = time.perf_counter()

#     # Чтение и декодирование изображения
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

#     # Обнаружение лиц на изображении
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

#     # Для простоты обрабатываем первое найденное лицо
#     face_data = detection_results[0]
#     x, y, width, height = face_data['box']
#     face_img = img[y:y+height, x:x+width]

#     # Получаем эмбеддинг с помощью keras-facenet
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     embedding = embeddings[0]

#     # Сохраняем эмбеддинг и данные пациента в базу данных
#     conn = await get_db_connection()
#     async with conn.cursor() as cursor:
#         await cursor.execute(
#             "INSERT INTO faces (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path) VALUES (%s, %s, %s, %s, %s, %s)",
#             (patient_id, hospital_id, branch_id, palata_id, file.filename, str(embedding.tolist()))
#         )
#         await conn.commit()
#     conn.close()

#     store_time = time.perf_counter() - start_time

#     return JSONResponse(content={
#         "status": True,
#         "message": "Эмбеддинг и данные пациента успешно сохранены.",
#         "store_time": store_time
#     })

# @app.post("/compare-face/")
# async def compare_face(file: UploadFile = File(...)):
#     """
#     API для сравнения эмбеддинга лица с сохранёнными в базе данных.
#     """
#     start_time = time.perf_counter()

#     # Чтение и декодирование изображения
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

#     # Обнаружение лиц
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

#     # Обрабатываем первое найденное лицо
#     face_data = detection_results[0]
#     x, y, width, height = face_data['box']
#     face_img = img[y:y+height, x:x+width]

#     # Получаем эмбеддинг для сравнения
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     query_embedding = embeddings[0]

#     # Получаем все эмбеддинги из базы данных
#     conn = await get_db_connection()
#     async with conn.cursor() as cursor:
#         await cursor.execute("SELECT patient_id, emb_path FROM faces")
#         stored_faces = await cursor.fetchall()
#     conn.close()

#     # Сравнение с сохранёнными эмбеддингами
#     best_similarity = -1.0
#     best_match_id = None

#     for record in stored_faces:
#         stored_embedding = np.array(eval(record[1]))  # Преобразуем строку обратно в массив
#         similarity = cosine_similarity(query_embedding, stored_embedding)
#         if similarity > best_similarity:
#             best_similarity = similarity
#             best_match_id = record[0]

#     compare_time = time.perf_counter() - start_time

#     # Приводим схожесть к процентному значению и преобразуем в стандартный float
#     match_percentage = float(best_similarity * 100)

#     if best_similarity >= SIMILARITY_THRESHOLD:
#         return JSONResponse(content={
#             "status": True,
#             "match_id": int(best_match_id),  # на всякий случай приводим к int
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

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)

# import time
# import cv2
# import numpy as np
# from mtcnn import MTCNN
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# from keras_facenet import FaceNet 

# app = FastAPI()

# # Инициализируем детектор лиц (MTCNN)
# detector = MTCNN()

# # Инициализируем объект FaceNet, модель будет загружена автоматически при первом вызове
# embedder = FaceNet()

# # Глобальное хранилище для сохранённых эмбеддингов (используем in-memory список)
# # Каждый элемент – словарь вида: {"id": int, "embedding": np.array, "timestamp": float}
# stored_faces = []
# current_id = 0  # Для генерации уникальных идентификаторов

# def cosine_similarity(a, b):
#     """Вычисление косинусной схожести между двумя векторами."""
#     a_norm = np.linalg.norm(a)
#     b_norm = np.linalg.norm(b)
#     if a_norm == 0 or b_norm == 0:
#         return 0.0
#     similarity = np.dot(a, b) / (a_norm * b_norm)
#     return similarity

# # Порог схожести для определения совпадения (значение подбирается экспериментально)
# SIMILARITY_THRESHOLD = 0.7

# @app.post("/store")
# async def store_face(file: UploadFile = File(...)):
#     """
#     API для сохранения эмбеддинга лица.
#     Принимает изображение, обнаруживает лицо, вычисляет эмбеддинг (с помощью keras-facenet) и сохраняет его.
#     Возвращает время, затраченное на обработку и сохранение.
#     """
#     global current_id
#     start_time = time.perf_counter()

#     # Чтение и декодирование изображения
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

#     # Обнаружение лиц на изображении
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

#     # Для простоты обрабатываем первое найденное лицо
#     face_data = detection_results[0]
#     x, y, width, height = face_data['box']
#     face_img = img[y:y+height, x:x+width]

#     # Получаем эмбеддинг с помощью keras-facenet
#     # Метод embeddings принимает список изображений (numpy-массивов или путей)
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     embedding = embeddings[0]

#     # Сохраняем эмбеддинг вместе с метаданными
#     stored_faces.append({
#         "id": current_id,
#         "embedding": embedding,
#         "timestamp": time.time()
#     })
#     current_id += 1

#     store_time = time.perf_counter() - start_time

#     return JSONResponse(content={
#         "status": True,
#         "message": "Эмбеддинг успешно сохранён.",
#         "store_time": store_time
#     })


# @app.post("/compare")
# async def compare_face(file: UploadFile = File(...)):
#     """
#     API для сравнения эмбеддинга лица с сохранёнными.
#     Принимает изображение, обнаруживает лицо, вычисляет эмбеддинг и сравнивает с сохранёнными эмбеддингами.
#     Если совпадение найдено (с учётом порога), возвращает status true, процент совпадения и время сравнения.
#     """
#     if not stored_faces:
#         raise HTTPException(status_code=404, detail="Нет сохранённых данных для сравнения.")

#     start_time = time.perf_counter()

#     # Чтение и декодирование изображения
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

#     # Обнаружение лиц
#     detection_results = detector.detect_faces(img)
#     if not detection_results:
#         raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

#     # Обрабатываем первое найденное лицо
#     face_data = detection_results[0]
#     x, y, width, height = face_data['box']
#     face_img = img[y:y+height, x:x+width]

#     # Получаем эмбеддинг для сравнения
#     embeddings = embedder.embeddings([face_img])
#     if embeddings.size == 0:
#         raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
#     query_embedding = embeddings[0]

#     # Сравнение с сохранёнными эмбеддингами
#     best_similarity = -1.0
#     best_match_id = None

#     for record in stored_faces:
#         stored_embedding = record["embedding"]
#         similarity = cosine_similarity(query_embedding, stored_embedding)
#         if similarity > best_similarity:
#             best_similarity = similarity
#             best_match_id = record["id"]

#     compare_time = time.perf_counter() - start_time

#     # Приводим схожесть к процентному значению и преобразуем в стандартный float
#     match_percentage = float(best_similarity * 100)

#     if best_similarity >= SIMILARITY_THRESHOLD:
#         return JSONResponse(content={
#             "status": True,
#             "match_id": int(best_match_id),  # на всякий случай приводим к int
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


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)

import time
import os
import uuid
import cv2
import numpy as np
import aiomysql
from mtcnn import MTCNN
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from keras_facenet import FaceNet

app = FastAPI()

# Конфигурация подключения к БД
DB_CONFIG = {
    "host": "face.tabet-kitap.kz",
    "user": "fastapi_user",
    "password": "secure_password",
    "database": "face_db",
    "port": 3306
}

# Создаём директории для хранения изображений и эмбеддингов (если их ещё нет)
os.makedirs("images", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# Инициализируем детектор лиц и модель для эмбеддингов
detector = MTCNN()
embedder = FaceNet()

# Порог схожести для определения совпадения
SIMILARITY_THRESHOLD = 0.7

def cosine_similarity(a, b):
    """Вычисление косинусной схожести между двумя векторами."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)

async def get_db_connection():
    """Асинхронное получение подключения к базе данных."""
    conn = await aiomysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        db=DB_CONFIG["database"],
        autocommit=True
    )
    return conn

@app.post("/store")
async def store_face(
    patient_id: str = Form(...),
    hospital_id: str = Form(...),
    branch_id: str = Form(...),
    palata_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Принимает изображение и дополнительные данные, 
    обнаруживает лицо, вычисляет эмбеддинг и сохраняет информацию в БД.
    """
    start_time = time.perf_counter()

    # Чтение изображения
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

    # Обнаружение лица
    detection_results = detector.detect_faces(img)
    if not detection_results:
        raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

    face_data = detection_results[0]
    x, y, width, height = face_data['box']
    face_img = img[y:y+height, x:x+width]

    # Вычисление эмбеддинга
    embeddings = embedder.embeddings([face_img])
    if embeddings.size == 0:
        raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
    embedding = embeddings[0]

    # Сохраняем изображение и эмбеддинг в файлы
    image_filename = f"images/{uuid.uuid4()}.jpg"
    embedding_filename = f"embeddings/{uuid.uuid4()}.npy"

    cv2.imwrite(image_filename, face_img)
    np.save(embedding_filename, embedding)

    # Сохраняем запись в базу данных
    try:
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            query = """
            INSERT INTO faces (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            await cursor.execute(query, (
                patient_id,
                hospital_id,
                branch_id,
                palata_id,
                image_filename,
                embedding_filename
            ))
            await conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в базу данных: {str(e)}")
    finally:
        conn.close()

    store_time = time.perf_counter() - start_time

    return JSONResponse(content={
        "status": True,
        "message": "Эмбеддинг успешно сохранён в базу данных.",
        "store_time": store_time
    })

@app.post("/compare")
async def compare_face(file: UploadFile = File(...)):
    """
    Принимает изображение, вычисляет эмбеддинг и сравнивает с данными из БД.
    Возвращает информацию о лучшем совпадении, если таковое найдено.
    """
    start_time = time.perf_counter()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")

    detection_results = detector.detect_faces(img)
    if not detection_results:
        raise HTTPException(status_code=404, detail="Лицо не обнаружено.")

    face_data = detection_results[0]
    x, y, width, height = face_data['box']
    face_img = img[y:y+height, x:x+width]

    embeddings = embedder.embeddings([face_img])
    if embeddings.size == 0:
        raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
    query_embedding = embeddings[0]

    # Извлекаем все записи из таблицы faces
    try:
        conn = await get_db_connection()
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("SELECT * FROM faces")
            rows = await cursor.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных из базы: {str(e)}")
    finally:
        conn.close()

    best_similarity = -1.0
    best_match = None

    for row in rows:
        emb_path = row["emb_path"]
        if not os.path.exists(emb_path):
            continue
        stored_embedding = np.load(emb_path)
        similarity = cosine_similarity(query_embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = row

    compare_time = time.perf_counter() - start_time
    match_percentage = float(best_similarity * 100)

    if best_similarity >= SIMILARITY_THRESHOLD and best_match is not None:
        return JSONResponse(content={
            "status": True,
            "match_details": best_match,
            "match_percentage": match_percentage,
            "compare_time": compare_time
        })
    else:
        return JSONResponse(content={
            "status": False,
            "message": "Совпадений не найдено.",
            "match_percentage": match_percentage,
            "compare_time": compare_time
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
