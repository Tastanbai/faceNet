import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Скрывает ненужные предупреждения

import uuid
import base64
import cv2
import numpy as np
import asyncio
import hashlib
from fastapi import HTTPException
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Инициализируем детектор и эмбеддер один раз
detector = MTCNN()
embedder = FaceNet()

# Асинхронная обёртка для чтения изображения из UploadFile
async def load_image_from_upload(file) -> np.ndarray:
    return await asyncio.to_thread(_load_image_from_upload, file)

def _load_image_from_upload(file) -> np.ndarray:
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Невозможно прочитать изображение.")
    return img

# Асинхронная обёртка для загрузки изображения из Base64-строки
async def load_image_from_base64(b64_string: str) -> np.ndarray:
    return await asyncio.to_thread(_load_image_from_base64, b64_string)

def _load_image_from_base64(b64_string: str) -> np.ndarray:
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

# Асинхронная функция детекции лица
async def detect_face(img: np.ndarray) -> np.ndarray:
    detection_results = await asyncio.to_thread(detector.detect_faces, img)
    if not detection_results:
        raise HTTPException(status_code=404, detail="Лицо не обнаружено.")
    face_data = detection_results[0]
    x, y, w, h = face_data['box']
    x, y = max(x, 0), max(y, 0)
    face_img = img[y:y+h, x:x+w]
    if face_img.size == 0:
        raise HTTPException(status_code=400, detail="Невозможно вырезать лицо.")
    return face_img

# Асинхронная функция вычисления эмбеддинга
async def get_embedding(face_img: np.ndarray) -> np.ndarray:
    embeddings = await asyncio.to_thread(embedder.embeddings, [face_img])
    if embeddings.size == 0:
        raise HTTPException(status_code=500, detail="Не удалось вычислить эмбеддинг.")
    return embeddings[0]

def calculate_hash(embedding: np.ndarray) -> str:
    return hashlib.sha256(embedding.tobytes()).hexdigest()

# Функция сохранения файлов (можно обернуть в asyncio.to_thread при необходимости)
async def save_face_and_embedding(face_img: np.ndarray, embedding: np.ndarray) -> (str, str):
    # Оборачивание операций записи в отдельные потоки
    image_filename = f"images/{uuid.uuid4()}.jpg"
    embedding_filename = f"embeddings/{uuid.uuid4()}.npy"
    await asyncio.to_thread(cv2.imwrite, image_filename, face_img)
    await asyncio.to_thread(np.save, embedding_filename, embedding)
    return image_filename, embedding_filename
