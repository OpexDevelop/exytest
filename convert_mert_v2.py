import sys
from unittest.mock import MagicMock

# Заглушка для torchaudio, который больше не выпускается под новый PyTorch 2.14+,
# но проверяется dynamic_module_utils при импорте кода MERT-v2 с Hugging Face
sys.modules["torchaudio"] = MagicMock()
sys.modules["torchaudio.functional"] = MagicMock()
sys.modules["torchaudio.transforms"] = MagicMock()

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import litert_torch as litert

class MERTv2FullSongWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v2-FullSong",
            trust_remote_code=True
        )
        self.mert.eval()

    def forward(self, input_values, attention_mask):
        # input_values: [1, N_SAMPLES], attention_mask: [1, N_SAMPLES] (1 - звук, 0 - паддинг)
        outputs = self.mert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        frames = outputs.last_hidden_state  # [1, frames, 1024] (25 Hz)
        
        # Штатное усреднение MERT-v2-FullSong по маске реальной длины песни
        mask = outputs.feature_attention_mask.unsqueeze(-1).to(frames.dtype)
        embedding = (frames * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        return F.normalize(embedding, p=2, dim=-1)

print("[1/3] Загрузка MERT-v2-FullSong (632M)...")
model = MERTv2FullSongWrapper().eval()

# Полный трек: 180 секунд (3 минуты) при 24 000 Гц = 4 320 000 сэмплов
# Если нужно 4 минуты — поставь 240 * 24000 = 5_760_000
FULL_SONG_SAMPLES = 180 * 24000

sample_waveform = torch.zeros(1, FULL_SONG_SAMPLES, dtype=torch.float32)
sample_mask = torch.ones(1, FULL_SONG_SAMPLES, dtype=torch.int32)

print(f"[2/3] Конвертация полного контекста ({FULL_SONG_SAMPLES} сэмплов) в LiteRT...")
edge_model = litert.convert(model, (sample_waveform, sample_mask))

output_name = "mert_v2_fullsong.tflite"
print(f"[3/3] Экспорт {output_name}...")
edge_model.export(output_name)
print(f"[✓] Готово: {output_name}")

