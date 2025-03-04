from tortoise import fields, models

class FaceNet(models.Model):
    id = fields.IntField(pk=True)
    patient_id = fields.CharField(max_length=255)
    hospital_id = fields.CharField(max_length=255)
    branch_id = fields.CharField(max_length=255)
    palata_id = fields.CharField(max_length=255)
    image_path = fields.CharField(max_length=255)
    emb_path = fields.CharField(max_length=255)
    emb_hash = fields.CharField(max_length=255, index=True)
    created_at = fields.DatetimeField(auto_now_add=True)

class FaceData(models.Model):
    id = fields.IntField(pk=True)
    patient_id = fields.CharField(max_length=255)
    hospital_id = fields.CharField(max_length=255)
    status = fields.BooleanField()
    similarity_percentage = fields.FloatField()
    comparison_time = fields.FloatField()
    timestamp = fields.DatetimeField(auto_now_add=True)
    is_delete = fields.BooleanField(default=False)

class QR(models.Model):
    id = fields.IntField(pk=True)
    status = fields.BooleanField()
    patient_id = fields.CharField(max_length=255)
    created_at = fields.DatetimeField(auto_now_add=True)
