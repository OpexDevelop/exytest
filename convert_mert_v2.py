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
# 2. ЭКСПОРТ MERT-v2-FullSong
# =====================================================================
from transformers import AutoModel
import litert_torch as litert

print("[1/3] Загрузка весов MERT-v2-FullSong (low_cpu_mem_usage)...")
raw_mert = AutoModel.from_pretrained(
    "m-a-p/MERT-v2-FullSong",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).eval()

class MERTFullSongExporter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mert = model

    def forward(self, input_values):
        # input_values: [1, samples] (сырая волна 24 кГц)
        out = self.mert(input_values=input_values, output_hidden_states=False)
        frames = out.last_hidden_state  # [1, frames, 1024]
        
        # Получаем единый вектор трека через усреднение по времени
        pooled = torch.mean(frames, dim=1)
        return F.normalize(pooled, p=2, dim=-1)

model_to_export = MERTFullSongExporter(raw_mert).eval()

# 180 секунд (3 минуты) при 24 кГц = 4 320 000 сэмплов
FULL_SONG_SAMPLES = 180 * 24000
sample_waveform = (torch.zeros(1, FULL_SONG_SAMPLES, dtype=torch.float32),)

print(f"[2/3] Конвертация графа (180 сек = {FULL_SONG_SAMPLES} сэмплов) в LiteRT...")
edge_model = litert.convert(model_to_export, sample_waveform)

output_filename = "mert_v2_fullsong.tflite"
print(f"[3/3] Экспорт {output_filename}...")
edge_model.export(output_filename)
print(f"[✓] УСПЕШНО! Модель создана: {output_filename}")

