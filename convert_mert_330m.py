import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import ai_edge_torch

class MERT330MChunkEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Загружаем тяжелый трансформер на 330 млн параметров (24 слоя)
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.mert.eval()

    def forward(self, waveform_chunk):
        # Вход: ровно 10 секунд аудио при 24 кГц -> [1, 240000]
        outputs = self.mert(waveform_chunk)
        
        # Берем 24-й слой (высшая семантика и гармония): [1, time_steps, 1024]
        last_hidden = outputs.last_hidden_state
        
        # Схлопываем время внутри чанка в один вектор [1, 1024]
        chunk_vector = torch.mean(last_hidden, dim=1)
        return chunk_vector

print("[1/3] Загрузка MERT-v1-330M...")
model = MERT330MChunkEmbedder().eval()

# 10 секунд при 24000 Гц = 240 000 сэмплов
sample_input = (torch.zeros(1, 240000, dtype=torch.float32),)

print("[2/3] Конвертация графа в LiteRT через ai-edge-torch...")
edge_model = ai_edge_torch.convert(model, sample_input)

print("[3/3] Экспорт TFLite...")
output_filename = "mert_330m_10s.tflite"
edge_model.export(output_filename)
print(f"[✓] Успешно создана модель: {output_filename}")
