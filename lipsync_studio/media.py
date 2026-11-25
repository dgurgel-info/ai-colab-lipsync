"""Operações de mídia com FFmpeg: inspeção, normalização, ajuste de duração, partes e montagem.

Regras que mantêm a sincronia exata:
* Todo vídeo é convertido para 25 quadros por segundo constantes (o InfiniteTalk trabalha a 25 fps).
* Partes, ajustes e montagem contam QUADROS, não segundos, então não acumulam deriva.
* O áudio final é o original do usuário (48 kHz), não o de 16 kHz que o modelo usa por dentro.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence

FPS = 25
MODEL_SAMPLE_RATE = 16000
MASTER_SAMPLE_RATE = 48000

# Qualidade das etapas intermediárias (quase sem perda) e do arquivo final.
CRF_WORK = 12
CRF_FINAL = 16


class MediaError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0            # quadros contados (0 quando o contêiner não informa)

    def describe_video(self) -> str:
        return f"{self.width}×{self.height} · {self.fps:.2f} fps · {format_time(self.duration)}"


# --------------------------------------------------------------------------- binários
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depende do ambiente
        raise MediaError("FFmpeg não encontrado. No Colab, execute o Passo 1 novamente.") from exc


def run_ffmpeg(args: Sequence[str], what: str = "FFmpeg") -> str:
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", *map(str, args)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        tail = "\n".join(res.stderr.strip().splitlines()[-12:])
        raise MediaError(f"{what} falhou (código {res.returncode}).\n{tail}")
    return res.stderr


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h:d}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


# --------------------------------------------------------------------------- inspeção
def _parse_rate(text: str) -> float:
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(text)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _probe_ffprobe(path: str) -> Optional[MediaInfo]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    res = subprocess.run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         errors="replace")
    if res.returncode != 0:
        raise MediaError(f"Não foi possível ler o arquivo: {os.path.basename(path)}\n{res.stderr.strip()[-400:]}")
    data = json.loads(res.stdout or "{}")
    info = MediaInfo(duration=float(data.get("format", {}).get("duration") or 0.0))
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and not info.has_video:
            if (st.get("disposition") or {}).get("attached_pic"):
                continue                      # capa de MP3 não é vídeo
            info.has_video = True
            info.width, info.height = int(st.get("width") or 0), int(st.get("height") or 0)
            rotation = _rotation(st)
            if rotation in (90, 270):
                info.width, info.height = info.height, info.width
            info.fps = _parse_rate(st.get("avg_frame_rate") or "") or _parse_rate(st.get("r_frame_rate") or "")
            info.frames = int(st.get("nb_frames") or 0)
            if not info.duration and st.get("duration"):
                info.duration = float(st["duration"])
        elif st.get("codec_type") == "audio":
            info.has_audio = True
            if not info.duration and st.get("duration"):
                info.duration = float(st["duration"])
    return info


def _rotation(stream: dict) -> int:
    rot = (stream.get("tags") or {}).get("rotate")
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rot = side["rotation"]
    try:
        return abs(int(float(rot))) % 360 if rot is not None else 0
    except ValueError:
        return 0


def _probe_ffmpeg(path: str) -> MediaInfo:
    """Sem ffprobe (ex.: binário do imageio-ffmpeg): lê o cabeçalho que o ``ffmpeg -i`` imprime."""
    res = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    text = res.stderr
    if "Invalid data found" in text or "No such file" in text:
        raise MediaError(f"Não foi possível ler o arquivo: {os.path.basename(path)}")
    info = MediaInfo()
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        info.duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    for line in text.splitlines():
        if "Stream #" in line and "Video:" in line and not info.has_video and "attached pic" not in line:
            info.has_video = True
            size = re.search(r",\s*(\d{2,5})x(\d{2,5})", line)
            if size:
                info.width, info.height = int(size.group(1)), int(size.group(2))
            rate = re.search(r"([\d.]+)\s*fps", line)
            if rate:
                info.fps = float(rate.group(1))
        elif "Stream #" in line and "Audio:" in line:
            info.has_audio = True
    if re.search(r"rotate\s*:\s*-?(90|270)", text) or re.search(r"rotation of -?(90|270)", text):
        info.width, info.height = info.height, info.width
    return info


def probe(path: str) -> MediaInfo:
    if not path or not os.path.exists(path):
        raise MediaError(f"Arquivo não encontrado: {path}")
    info = _probe_ffprobe(path)
    return info if info is not None else _probe_ffmpeg(path)


def count_frames(path: str) -> int:
    """Conta os quadros decodificando (exato; usado nos arquivos que o próprio Studio gera)."""
    text = run_ffmpeg(["-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"], "Contagem de quadros")
    found = re.findall(r"frame=\s*(\d+)", text)
    if found:
        return int(found[-1])
    info = probe(path)
    return info.frames or int(round(info.duration * FPS))


# --------------------------------------------------------------------------- normalização
def _scale_filter(max_side: Optional[int]) -> str:
    """Limita o lado MENOR a ``max_side`` (vale para vídeo deitado e em pé) e garante dimensões pares."""
    if not max_side:
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
    m = int(max_side)
    w = f"'if(gt(iw,ih),-2,trunc(min({m},iw)/2)*2)'"
    h = f"'if(gt(iw,ih),trunc(min({m},ih)/2)*2,-2)'"
    return f"scale={w}:{h}:flags=lanczos"


def _x264(crf: int = CRF_WORK, preset: str = "medium") -> List[str]:
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


def normalize_video(src: str, dst: str, max_side: Optional[int] = 1080) -> MediaInfo:
    """Vídeo sem áudio, 25 fps constantes, H.264 yuv420p, lado menor limitado a ``max_side``."""
    info = probe(src)
    if not info.has_video:
        raise MediaError("O arquivo de vídeo não tem trilha de imagem.")
    vf = f"fps={FPS},{_scale_filter(max_side)},setsar=1,format=yuv420p"
    run_ffmpeg(["-i", src, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, *_x264(),
                "-movflags", "+faststart", dst], "Conversão do vídeo")
    out = probe(dst)
    out.frames = count_frames(dst)
    return out


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def is_image(path: str) -> bool:
    return os.path.splitext(path or "")[1].lower() in IMAGE_EXTENSIONS


def image_to_video(src: str, dst: str, frames: int, max_side: Optional[int] = 1080) -> MediaInfo:
    """Foto -> vídeo parado com ``frames`` quadros (o InfiniteTalk anima a pessoa a partir dele)."""
    vf = f"{_scale_filter(max_side)},setsar=1,fps={FPS},format=yuv420p"
    run_ffmpeg(["-loop", "1", "-i", src, "-vf", vf, "-frames:v", str(frames), *_x264(), "-an",
                "-movflags", "+faststart", dst], "Conversão da foto em vídeo")
    out = probe(dst)
    out.frames = count_frames(dst)
    return out


def normalize_audio(src: str, dst: str, sample_rate: int = MASTER_SAMPLE_RATE, channels: int = 2) -> MediaInfo:
    """Extrai/converte o áudio para WAV PCM (aceita arquivo de áudio ou de vídeo)."""
    info = probe(src)
    if not info.has_audio:
        raise MediaError("O arquivo de áudio não tem som.")
    run_ffmpeg(["-i", src, "-map", "0:a:0", "-vn", "-ac", str(channels), "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", dst], "Conversão do áudio")
    return probe(dst)


def fit_audio(src: str, dst: str, frames: int) -> None:
    """Corta ou completa com silêncio até exatamente ``frames`` quadros de duração."""
    seconds = frames / FPS
    run_ffmpeg(["-i", src, "-af", f"apad=whole_dur={seconds:.6f}", "-t", f"{seconds:.6f}",
                "-c:a", "pcm_s16le", dst], "Ajuste da duração do áudio")


def cut_audio(src: str, dst: str, start_frame: int, frames: int, sample_rate: int = MODEL_SAMPLE_RATE) -> None:
    start, seconds = start_frame / FPS, frames / FPS
    run_ffmpeg(["-i", src, "-ss", f"{start:.6f}", "-t", f"{seconds:.6f}",
                "-af", f"apad=whole_dur={seconds:.6f}", "-ac", "1", "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", dst], "Corte do áudio")


def cut_video(src: str, dst: str, start_frame: int, frames: int) -> None:
    """Recorte exato por quadros (o vídeo de origem já está a 25 fps constantes)."""
    run_ffmpeg(["-ss", f"{start_frame / FPS:.6f}", "-i", src, "-map", "0:v:0", "-an",
                "-frames:v", str(frames), *_x264(), dst], "Corte do vídeo")


def conform_frames(src: str, dst: str, frames: int) -> None:
    """Garante exatamente ``frames`` quadros (corta o excesso ou repete o último quadro)."""
    run_ffmpeg(["-i", src, "-map", "0:v:0", "-an",
                "-vf", f"fps={FPS},tpad=stop_mode=clone:stop={FPS * 4},format=yuv420p",
                "-frames:v", str(frames), *_x264(), dst], "Ajuste de quadros")


def _concat_list(paths: Sequence[str], list_path: str) -> None:
    with open(list_path, "w", encoding="utf-8") as fh:
        for p in paths:
            safe = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            fh.write(f"file '{safe}'\n")


def concat_videos(paths: Sequence[str], dst: str, reencode: bool = False) -> None:
    """Junta vídeos com os mesmos parâmetros de codificação (as partes geradas pelo Studio)."""
    if not paths:
        raise MediaError("Nenhuma parte para juntar.")
    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="concat_")
    os.close(fd)
    try:
        _concat_list(paths, list_path)
        codec = _x264() if reencode else ["-c", "copy"]
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_path, "-map", "0:v:0", "-an", *codec, dst],
                   "Junção das partes")
    finally:
        os.remove(list_path)


def reverse_video(src: str, dst: str, chunk_frames: int = FPS * 4) -> None:
    """Inverte o vídeo em blocos curtos (o filtro reverse guarda tudo na memória)."""
    total = count_frames(src)
    work = tempfile.mkdtemp(prefix="reverse_", dir=os.path.dirname(os.path.abspath(dst)))
    try:
        pieces = []
        for i, start in enumerate(range(0, total, chunk_frames)):
            n = min(chunk_frames, total - start)
            piece = os.path.join(work, f"r{i:05d}.mp4")
            run_ffmpeg(["-ss", f"{start / FPS:.6f}", "-i", src, "-map", "0:v:0", "-an",
                        "-frames:v", str(n), "-vf", "reverse", *_x264(), piece], "Inversão do vídeo")
            pieces.append(piece)
        concat_videos(list(reversed(pieces)), dst)
    finally:
        shutil.rmtree(work, ignore_errors=True)


AUDIO_LONGER_MODES = ("pingpong", "loop", "trim")    # vai e volta | repete do início | corta o áudio
VIDEO_LONGER_MODES = ("trim", "silence")              # corta o vídeo | completa o áudio com silêncio


def fit_video(src: str, dst: str, frames: int, mode: str = "pingpong") -> None:
    """Deixa o vídeo (25 fps) com exatamente ``frames`` quadros.

    Mais curto que o necessário: ``pingpong`` (vai e volta, sem saltos) ou ``loop`` (recomeça do início).
    Mais longo: corta o final.
    """
    total = count_frames(src)
    if total <= 0:
        raise MediaError("O vídeo não tem quadros.")
    if frames <= total:
        if frames == total:
            shutil.copyfile(src, dst)
        else:
            run_ffmpeg(["-i", src, "-map", "0:v:0", "-frames:v", str(frames), *_x264(), dst], "Corte do vídeo")
        return
    if mode == "loop":
        run_ffmpeg(["-stream_loop", "-1", "-i", src, "-map", "0:v:0", "-frames:v", str(frames), *_x264(), dst],
                   "Repetição do vídeo")
        return
    work = tempfile.mkdtemp(prefix="pingpong_", dir=os.path.dirname(os.path.abspath(dst)))
    try:
        rev = os.path.join(work, "reverse.mp4")
        reverse_video(src, rev)
        cycles = math.ceil(frames / total)
        sequence = [src if i % 2 == 0 else rev for i in range(cycles)]
        joined = os.path.join(work, "joined.mp4")
        concat_videos(sequence, joined, reencode=True)
        run_ffmpeg(["-i", joined, "-map", "0:v:0", "-frames:v", str(frames), *_x264(), dst], "Vai e volta")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def preview_with_audio(video: str, audio: str, dst: str) -> str:
    """Prévia rápida de uma parte (imagem copiada, áudio em AAC) para ouvir a sincronia na revisão."""
    run_ffmpeg(["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "160k", "-shortest", "-movflags", "+faststart", dst], "Prévia da parte")
    return dst


def mux(video: str, audio: str, dst: str, background: Optional[str] = None, background_volume: float = 0.0,
        crf: int = CRF_FINAL) -> None:
    """Vídeo final: imagem sincronizada + áudio original (AAC 192 kbps), com som de fundo opcional."""
    args = ["-i", video, "-i", audio]
    use_bg = bool(background) and background_volume > 0
    if use_bg:
        args += ["-i", background]
        args += ["-filter_complex",
                 f"[2:a]aresample={MASTER_SAMPLE_RATE},volume={background_volume:.3f}[bg];"
                 f"[1:a][bg]amix=inputs=2:duration=first:normalize=0[a]",
                 "-map", "0:v:0", "-map", "[a]"]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    args += [*_x264(crf, "slow"), "-c:a", "aac", "-b:a", "192k", "-ar", str(MASTER_SAMPLE_RATE),
             "-shortest", "-movflags", "+faststart", dst]
    run_ffmpeg(args, "Montagem do vídeo final")
