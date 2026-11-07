import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

try:
    import litert_torch as litert
except ImportError:
    import ai_edge_torch as litert

class MERTv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Загружаем MERT-v2 без квантования
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True)
        self.mert.eval()

    def forward(self, waveform):
        # waveform: [1, 720000] (30 секунд сырого моно 24 кГц)
        outputs = self.mert(input_values=waveform)
        frames = outputs.last_hidden_state  # [1, 750, 1024]
        
        # Среднее по времени (750 токенов -> 1 вектор)
        pooled = torch.mean(frames, dim=1)  # [1, 1024]
        return F.normalize(pooled, p=2, dim=-1)

print("[*] Загрузка MERT-v2 весов в RAM...")
model = MERTv2Encoder().eval()

# 30 секунд аудио при 24 кГц = 720 000 сэмплов
sample_input = (torch.zeros(1, 720000, dtype=torch.float32),)

print("[*] Трассировка и экспорт графа в LiteRT (Float32, без квантования)...")
edge_model = litert.convert(model, sample_input)

output_name = "mert_v2_30s.tflite"
edge_model.export(output_name)
print(f"[✓] Модель успешно собрана: {output_name}")

