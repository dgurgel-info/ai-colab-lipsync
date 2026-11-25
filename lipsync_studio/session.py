"""Sessão de dublagem com o InfiniteTalk: configurações, partes e arquivos, salvos em ``sessoes/<id>/session.json``.

Cada parte é um trecho de vídeo + áudio processado separadamente pelo modelo. Assim um vídeo longo
não estoura a memória, e uma parte ruim pode ser refeita sem refazer o resto.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from lipsync_studio.media import FPS, format_time

PENDING, RUNNING, DONE, ERROR = "pending", "running", "done", "error"

STATUS_LABELS = {
    PENDING: "⏳ Pendente",
    RUNNING: "⚙️ Gerando",
    DONE: "✅ Pronta",
    ERROR: "❌ Erro",
}

DEFAULT_PROMPT = "A person is talking naturally to the camera, with natural facial expressions and head movements."


@dataclass
class Settings:
    prompt: str = DEFAULT_PROMPT    # descrição da cena (em inglês funciona melhor)
    resolution: str = "480"         # 480 ou 720 (o InfiniteTalk gera o quadro inteiro nessa resolução)
    steps: int = 40                 # passos de amostragem (o oficial usa 40)
    text_guide: float = 5.0         # orientação pelo texto (oficial: 5)
    audio_guide: float = 4.0        # orientação pelo áudio: mais alto = boca mais fiel (oficial: 3 a 5)
    seed: int = 42                  # mesma semente em todas as partes = resultado consistente
    teacache: bool = True           # acelera ~2x reaproveitando passos parecidos
    part_seconds: int = 30          # tamanho de cada parte
    audio_longer: str = "pingpong"  # áudio mais longo que o vídeo: pingpong | loop | trim
    video_longer: str = "trim"      # vídeo mais longo que o áudio: trim | silence
    background_volume: float = 0.0  # volume do som original do vídeo ao fundo (0 = desligado)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Settings":
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def validate(self) -> "Settings":
        self.prompt = (self.prompt or "").strip() or DEFAULT_PROMPT
        self.resolution = "720" if str(self.resolution).startswith("720") else "480"
        self.steps = int(min(50, max(4, int(self.steps))))
        self.text_guide = float(min(10.0, max(1.0, float(self.text_guide))))
        self.audio_guide = float(min(10.0, max(1.0, float(self.audio_guide))))
        self.seed = int(self.seed) if int(self.seed) >= 0 else 42
        self.part_seconds = int(min(90, max(5, int(self.part_seconds))))
        self.background_volume = float(min(1.0, max(0.0, float(self.background_volume))))
        if self.audio_longer not in ("pingpong", "loop", "trim"):
            self.audio_longer = "pingpong"
        if self.video_longer not in ("trim", "silence"):
            self.video_longer = "trim"
        return self


@dataclass
class Part:
    index: int                      # começa em 1
    start_frame: int
    frames: int
    status: str = PENDING
    seed: Optional[int] = None      # semente usada na última geração
    steps: Optional[int] = None
    audio_guide: Optional[float] = None
    seconds: float = 0.0            # tempo de processamento da última geração
    attempts: int = 0
    message: str = ""

    @property
    def start(self) -> float:
        return self.start_frame / FPS

    @property
    def end(self) -> float:
        return (self.start_frame + self.frames) / FPS

    @property
    def name(self) -> str:
        return f"p{self.index:03d}"

    def label(self) -> str:
        return f"Parte {self.index} · {format_time(self.start)} – {format_time(self.end)}"


def plan_parts(total_frames: int, part_frames: int) -> List[Part]:
    """Divide ``total_frames`` em partes de tamanho parecido (nenhuma parte minúscula no final)."""
    if total_frames <= 0:
        return []
    part_frames = max(1, int(part_frames))
    count = max(1, int(round(total_frames / part_frames)))
    bounds = [round(i * total_frames / count) for i in range(count + 1)]
    return [Part(index=i + 1, start_frame=bounds[i], frames=bounds[i + 1] - bounds[i]) for i in range(count)]


def _slug(text: str, limit: int = 40) -> str:
    base = os.path.splitext(os.path.basename(text or ""))[0]
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    return (base or "video")[:limit]


@dataclass
class Session:
    id: str
    dir: str
    title: str
    created: float
    settings: Settings
    total_frames: int = 0
    width: int = 0
    height: int = 0
    source_video: str = ""          # nomes originais (só para exibir)
    source_kind: str = "video"      # video | image
    source_audio: str = ""
    video_frames: int = 0           # duração original do vídeo (quadros a 25 fps)
    audio_frames: int = 0           # duração original do áudio (quadros a 25 fps)
    has_background: bool = False
    parts: List[Part] = field(default_factory=list)
    output: str = ""                # caminho do último vídeo final montado

    # ------------------------------------------------------------------ criação e disco
    @classmethod
    def create(cls, root: str, video_name: str, audio_name: str, settings: Settings) -> "Session":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sid = f"{stamp}_{_slug(video_name, 24)}_{uuid.uuid4().hex[:4]}"
        path = os.path.abspath(os.path.join(root, sid))
        os.makedirs(os.path.join(path, "src"), exist_ok=True)
        os.makedirs(os.path.join(path, "parts"), exist_ok=True)
        return cls(id=sid, dir=path, title=_slug(video_name), created=time.time(), settings=settings,
                   source_video=os.path.basename(video_name or ""), source_audio=os.path.basename(audio_name or ""))

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("dir")
        return data

    def save(self) -> None:
        path = os.path.join(self.dir, "session.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Session":
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        with open(os.path.join(folder, "session.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        parts = [Part(**p) for p in data.pop("parts", [])]
        settings = Settings.from_dict(data.pop("settings", {}))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(dir=os.path.abspath(folder), settings=settings, parts=parts, **known)

    # ------------------------------------------------------------------ caminhos
    def path(self, *names: str) -> str:
        return os.path.join(self.dir, *names)

    @property
    def video_path(self) -> str:
        return self.path("src", "video.mp4")      # 25 fps, já com a duração final (foto vira vídeo parado)

    @property
    def audio_path(self) -> str:
        return self.path("src", "audio.wav")      # áudio original 48 kHz, com a duração final

    @property
    def model_audio_path(self) -> str:
        return self.path("src", "audio16k.wav")   # mono 16 kHz para o modelo

    @property
    def background_path(self) -> str:
        return self.path("src", "background.wav")

    def part_dir(self, part: Part) -> str:
        return self.path("parts", part.name)

    def part_input_video(self, part: Part) -> str:
        return os.path.join(self.part_dir(part), "input.mp4")

    def part_input_audio(self, part: Part) -> str:
        return os.path.join(self.part_dir(part), "input.wav")

    def part_output(self, part: Part) -> str:
        return os.path.join(self.part_dir(part), "infinitetalk.mp4")

    # ------------------------------------------------------------------ consultas
    def get_part(self, index: int) -> Part:
        for part in self.parts:
            if part.index == int(index):
                return part
        raise KeyError(f"Parte {index} não existe nesta sessão.")

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in STATUS_LABELS}
        for part in self.parts:
            out[part.status] = out.get(part.status, 0) + 1
        return out

    def pending_parts(self) -> List[Part]:
        return [p for p in self.parts if p.status in (PENDING, RUNNING, ERROR)]

    def ready_to_assemble(self) -> bool:
        return bool(self.parts) and all(p.status == DONE for p in self.parts)

    @property
    def duration(self) -> float:
        return self.total_frames / FPS

    def display_name(self) -> str:
        when = time.strftime("%d/%m %H:%M", time.localtime(self.created))
        done = self.counts()[DONE]
        return f"{when} · {self.title} · {format_time(self.duration)} · {done}/{len(self.parts)} partes"


def list_sessions(root: str) -> List[Session]:
    """Sessões salvas, da mais nova para a mais antiga (ignora pastas incompletas)."""
    found = []
    if not os.path.isdir(root):
        return found
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if os.path.isfile(os.path.join(folder, "session.json")):
            try:
                found.append(Session.load(folder))
            except (OSError, ValueError, TypeError):
                continue
    found.sort(key=lambda s: s.created, reverse=True)
    return found
