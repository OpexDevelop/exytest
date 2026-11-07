import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import ai_edge_torch

class MERTv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Загружаем MERT-v2 без квантования
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True)
        self.mert.eval()

    def forward(self, waveform):
        # waveform: [1, 720000] (30 секунд сырого моно 24 кГц)
        outputs = self.mert(waveform)
        frames = outputs.last_hidden_state # [1, 750, 1024]
        
        # Получаем итоговый вектор трека через усреднение по времени
        pooled = torch.mean(frames, dim=1) # [1, 1024]
        return F.normalize(pooled, p=2, dim=-1)

print("[*] Загрузка MERT-v2 весов в памяти...")
model = MERTv2Encoder().eval()

# 30 секунд аудио при 24 кГц
sample_input = (torch.zeros(1, 720000, dtype=torch.float32),)

print("[*] Экспорт графа в чистый FP16...")
# Конвертируем без квантования: веса переводятся строго в float16 для GPU
edge_model = ai_edge_torch.convert(
    model, 
    sample_input,
    _ai_edge_converter_flags={"target_data_type": "float16"}
)

output_name = "mert_v2_fp16.tflite"
edge_model.export(output_name)
print(f"[✓] Модель сохранена: {output_name}")

