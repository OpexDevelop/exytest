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
# 2. ЗАГРУЗКА В FP16 C ОГРАНИЧЕНИЕМ ПАМЯТИ
# =====================================================================
from transformers import AutoModel
import litert_torch as litert

print("[1/3] Загрузка весов MERT-v2-FullSong (FP16, low_cpu_mem_usage)...")
raw_mert = AutoModel.from_pretrained(
    "m-a-p/MERT-v2-FullSong",
    trust_remote_code=True,
    torch_dtype=torch.float32,     # Для экспорта графа нужен float32
    low_cpu_mem_usage=True        # Загрузка без дублирования буферов в RAM
).eval()

print("[2/3] Подготовка трансформера к экспорту...")
print("Доступные модули в модели:", [name for name, _ in raw_mert.named_children()])

class MERTBackboneWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # Если есть вложенный трансформер (encoder/model) — берем его, иначе саму модель
        if hasattr(base_model, "encoder"):
            self.backbone = base_model.encoder
        elif hasattr(base_model, "model"):
            self.backbone = base_model.model
        else:
            self.backbone = base_model

    def forward(self, hidden_states):
        # hidden_states: [1, seq_len, 1024]
        out = self.backbone(hidden_states)
        
        # Обработка разных типов возвращаемых объектов (tuple или BaseModelOutput)
        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state
        elif isinstance(out, tuple):
            feats = out[0]
        else:
            feats = out

        # Усреднение по временным токенам -> единый вектор трека [1, 1024]
        pooled = torch.mean(feats, dim=1)
        return F.normalize(pooled, p=2, dim=-1)

model_to_export = MERTBackboneWrapper(raw_mert).eval()

# 3 минуты трека = 180 сек * 25 Hz = 4 500 токенов
SEQ_LEN = 180 * 25
sample_input = (torch.zeros(1, SEQ_LEN, 1024, dtype=torch.float32),)

print(f"[3/3] Конвертация LiteRT (последовательность {SEQ_LEN} токенов, FP32)...")
edge_model = litert.convert(model_to_export, sample_input)

output_filename = "mert_v2_backbone.tflite"
edge_model.export(output_filename)
print(f"[✓] УСПЕШНО! Модель создана: {output_filename}")

