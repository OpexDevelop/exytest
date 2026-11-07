import sys
import types
import importlib.machinery

# Полноценная заглушка со спецификацией модуля для Python 3.11+ и transformers
def register_dummy_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__version__ = "2.6.0"
    mod.__path__ = []
    sys.modules[name] = mod
    return mod

ta = register_dummy_module("torchaudio")
ta.functional = register_dummy_module("torchaudio.functional")
ta.transforms = register_dummy_module("torchaudio.transforms")
ta.compliance = register_dummy_module("torchaudio.compliance")
ta.compliance.kaldi = register_dummy_module("torchaudio.compliance.kaldi")

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
        outputs = self.mert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        frames = outputs.last_hidden_state  # [1, frames, 1024]
        
        mask = outputs.feature_attention_mask.unsqueeze(-1).to(frames.dtype)
        embedding = (frames * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        return F.normalize(embedding, p=2, dim=-1)

print("[1/3] Загрузка MERT-v2-FullSong (632M)...")
model = MERTv2FullSongWrapper().eval()

# 180 секунд (3 минуты) при 24 000 Гц = 4 320 000 сэмплов
FULL_SONG_SAMPLES = 180 * 24000

sample_waveform = torch.zeros(1, FULL_SONG_SAMPLES, dtype=torch.float32)
sample_mask = torch.ones(1, FULL_SONG_SAMPLES, dtype=torch.int32)

print(f"[2/3] Конвертация полного контекста ({FULL_SONG_SAMPLES} сэмплов) в LiteRT...")
edge_model = litert.convert(model, (sample_waveform, sample_mask))

output_name = "mert_v2_fullsong.tflite"
print(f"[3/3] Экспорт {output_name}...")
edge_model.export(output_name)
print(f"[✓] Готово: {output_name}")

