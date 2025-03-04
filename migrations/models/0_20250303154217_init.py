from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `facedata` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `patient_id` VARCHAR(255) NOT NULL,
    `hospital_id` VARCHAR(255) NOT NULL,
    `status` BOOL NOT NULL,
    `similarity_percentage` DOUBLE NOT NULL,
    `comparison_time` DOUBLE NOT NULL,
    `timestamp` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `is_delete` BOOL NOT NULL DEFAULT 0
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `facenet` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `patient_id` VARCHAR(255) NOT NULL,
    `hospital_id` VARCHAR(255) NOT NULL,
    `branch_id` VARCHAR(255) NOT NULL,
    `palata_id` VARCHAR(255) NOT NULL,
    `image_path` VARCHAR(255) NOT NULL,
    `emb_path` VARCHAR(255) NOT NULL,
    `emb_hash` VARCHAR(255) NOT NULL,
    `created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY `idx_facenet_emb_has_fb78ac` (`emb_hash`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `qr` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `status` BOOL NOT NULL,
    `patient_id` VARCHAR(255) NOT NULL,
    `created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `aerich` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `version` VARCHAR(255) NOT NULL,
    `app` VARCHAR(100) NOT NULL,
    `content` JSON NOT NULL
) CHARACTER SET utf8mb4;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
