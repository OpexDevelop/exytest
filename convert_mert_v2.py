import sys
import types
import importlib.machinery
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. ЧИСТАЯ РЕАЛИЗАЦИЯ ТРЕХ КЛАССОВ TORCHAUDIO НА PYTORCH
# =====================================================================
class DummySpectrogram(nn.Module):
    def __init__(self, n_fft=400, win_length=None, hop_length=None, pad=0, power=2.0, normalized=False, center=True, pad_mode="reflect", onesided=True):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, x):
        window = self.window.to(dtype=x.dtype, device=x.device)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True
        )
        if self.power is not None:
            return torch.abs(stft).pow(self.power)
        return stft

class DummyMelScale(nn.Module):
    def __init__(self, n_mels=128, sample_rate=16000, f_min=0.0, f_max=None, n_stft=201, norm=None, mel_scale="htk"):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.n_stft = n_stft
        # Простая инициализация фильтр-банка
        fb = torch.zeros(n_stft, n_mels)
        self.register_buffer("fb", fb)

    def forward(self, specgram):
        return torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

class DummyAmplitudeToDB(nn.Module):
    def __init__(self, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0):
        super().__init__()
        self.multiplier = multiplier
        self.amin = amin
        self.top_db = top_db

    def forward(self, x):
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        if self.top_db is not None:
            x_db = torch.maximum(x_db, x_db.max() - self.top_db)
        return x_db

# Регистрируем модуль в sys.modules чтобы huggingface-код MERT увидел torchaudio.transforms
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
# 2. ЗАГРУЗКА И ЭКСПОРТ MERT-V2
# =====================================================================
from transformers import AutoModel
import litert_torch as litert

print("[1/3] Загрузка модели MERT-v2-FullSong с Hugging Face...")
raw_mert = AutoModel.from_pretrained("m-a-p/MERT-v2-FullSong", trust_remote_code=True).eval()

print("[2/3] Подготовка трансформерного бэкенда для GPU инференса...")

class MERTBackboneOnly(nn.Module):
    def __init__(self, mert_model):
        super().__init__()
        self.encoder = mert_model.encoder

    def forward(self, hidden_features):
        # hidden_features: [1, seq_len, 1024]
        encoder_out = self.encoder(hidden_features)
        last_hidden = encoder_out.last_hidden_state  # [1, seq_len, 1024]
        
        # Mean pooling и L2-нормализация
        pooled = torch.mean(last_hidden, dim=1)
        return F.normalize(pooled, p=2, dim=-1)

backbone = MERTBackboneOnly(raw_mert).eval()

# 3 минуты аудио = 180 сек * 25 Hz = 4 500 токенов
SEQ_LEN = 180 * 25
sample_features = (torch.zeros(1, SEQ_LEN, 1024, dtype=torch.float32),)

print(f"[3/3] Конвертация графа (последовательность {SEQ_LEN} токенов, 1024 dim, FP32) в LiteRT...")
edge_model = litert.convert(backbone, sample_features)

output_filename = "mert_v2_backbone.tflite"
edge_model.export(output_filename)
print(f"[✓] Успешно экспортировано: {output_filename}")

