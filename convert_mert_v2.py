import sys
import types
import importlib.machinery
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. ЗАГЛУШКА TORCHAUDIO TRANSFORMS
# =====================================================================
class DummySpectrogram(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x): return x

class DummyMelScale(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x): return x

class DummyAmplitudeToDB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x): return x

def make_module(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    return m

mod_ta = make_module("torchaudio")
mod_tat = make_module("torchaudio.transforms")
mod_tat.Spectrogram = DummySpectrogram
mod_tat.MelScale = DummyMelScale
mod_tat.AmplitudeToDB = DummyAmplitudeToDB
mod_ta.transforms = mod_tat
sys.modules["torchaudio"] = mod_ta
sys.modules["torchaudio.transforms"] = mod_tat

# =====================================================================
# 2. ЗАГРУЗКА И ПАТЧ МОДЕЛИ MERT-v2
# =====================================================================
from transformers import AutoModel
import litert_torch as litert

print("[1/3] Загрузка MERT-v2-FullSong...")
raw_mert = AutoModel.from_pretrained(
    "m-a-p/MERT-v2-FullSong",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).eval()

# Патчим modeling_mert2: убираем data-dependent проверку if not isfinite
# Получаем оригинальный forward класса модели
model_cls = raw_mert.__class__
orig_forward = model_cls.forward

# Патчим инспекцию файла/модуля, либо заменяем forward без проверки
def patched_forward(self, input_values, attention_mask=None, **kwargs):
    # Убираем строку: if not torch.isfinite(input_values).all(): raise ...
    # Вызываем логику модели в обход проверки тензора
    return orig_forward(self, input_values=input_values, attention_mask=attention_mask, **kwargs)

# На случай если проверка сидит прямо внутри orig_forward, 
# перехватываем torch.isfinite во время трассировки:
orig_isfinite = torch.isfinite
def dummy_isfinite(tensor):
    # Возвращаем константный True тензор, чтобы .all() не создавал SymNode Guard
    return torch.ones_like(tensor, dtype=torch.bool)

torch.isfinite = dummy_isfinite

class MERTFullSongExporter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mert = model

    def forward(self, input_values):
        out = self.mert(input_values=input_values, output_hidden_states=False)
        frames = out.last_hidden_state  # [1, frames, 1024]
        pooled = torch.mean(frames, dim=1)
        return F.normalize(pooled, p=2, dim=-1)

model_to_export = MERTFullSongExporter(raw_mert).eval()

# 180 секунд = 4 320 000 сэмплов
FULL_SONG_SAMPLES = 180 * 24000
sample_waveform = (torch.zeros(1, FULL_SONG_SAMPLES, dtype=torch.float32),)

print(f"[2/3] Конвертация графа ({FULL_SONG_SAMPLES} сэмплов) в LiteRT...")
edge_model = litert.convert(model_to_export, sample_waveform)

# Восстанавливаем функцию после трассировки
torch.isfinite = orig_isfinite

output_filename = "mert_v2_fullsong.tflite"
print(f"[3/3] Экспорт {output_filename}...")
edge_model.export(output_filename)
print(f"[✓] УСПЕШНО! Модель создана: {output_filename}")

