TORTOISE_ORM = {
    "connections": {
        "default": "mysql://fastapi_user:secure_password@facenet.tabet-kitap.kz:3306/face_db"
    },
    "apps": {
        "models": {
            "models": ["models", "aerich.models"],  # Добавляем aerich.models для поддержки миграций
            "default_connection": "default",
        }
    },
}
