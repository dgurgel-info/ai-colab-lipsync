"""Tema, folha de estilo e trechos de HTML da interface (mesmo design system do Matraca Studio)."""

from __future__ import annotations

import html
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_css() -> str:
    with open(os.path.join(_HERE, "style.css"), encoding="utf-8") as fh:
        return fh.read()


def build_theme(gr):
    """Tema base do Gradio alinhado aos tokens do CSS (emerald/blue, cantos suaves)."""
    try:
        return gr.themes.Base(
            primary_hue="emerald", secondary_hue="blue", neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_md,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        )
    except Exception:  # versões antigas sem todos os parâmetros
        return gr.themes.Soft(primary_hue="emerald", secondary_hue="blue")


def _gradio_major(gr) -> int:
    try:
        return int(str(gr.__version__).split(".")[0])
    except Exception:
        return 5


def style_kwargs(gr, theme, css: str) -> tuple:
    """(kwargs do ``gr.Blocks``, kwargs do ``launch``). No Gradio 6 tema e CSS foram para o ``launch()``."""
    if _gradio_major(gr) >= 6:
        return {}, {"theme": theme, "css": css}
    return {"theme": theme, "css": css}, {}


ENGINE_STATES = {
    "ready": ("", "pronto"),
    "busy": ("", "gerando"),
    "loading": ("warn", "carregando"),
    "installing": ("warn", "instalando"),
    "stopped": ("warn", "parado"),
    "error": ("err", "com erro"),
}


def render_header(gpu_label: Optional[str], engine_state: str = "ready", version: str = "") -> str:
    pills = []
    if gpu_label:
        pills.append(f"<span class='ls-pill'><span class='ls-dot'></span><b>{html.escape(gpu_label)}</b></span>")
    else:
        pills.append("<span class='ls-pill warn'><span class='ls-dot'></span>GPU não detectada</span>")
    cls, text = ENGINE_STATES.get(engine_state, ("warn", engine_state))
    pills.append(f"<span class='ls-pill {cls}'><span class='ls-dot'></span>InfiniteTalk <b>{text}</b></span>")
    if version:
        pills.append(f"<span class='ls-pill'>v<b>{html.escape(version)}</b></span>")
    return (
        "<div class='ls-header'>"
        "<div class='ls-brand'><div class='ls-logo'>🎬</div>"
        "<div><h1 class='ls-title'>LipSync Studio</h1>"
        "<p class='ls-subtitle'>Dublagem de vídeo com InfiniteTalk · boca, rosto e corpo no ritmo da fala</p>"
        "</div></div>"
        f"<div class='ls-pills'>{''.join(pills)}</div></div>"
    )


def render_engine_progress(pct: Optional[float] = None, label: str = "", detail: str = "",
                           state: str = "running") -> str:
    """Barra fixa no topo enquanto o modelo instala ou carrega. ``pct=None`` esconde a barra."""
    if pct is None:
        return ""
    value = max(0, min(100, int(round(pct))))
    detail_html = f"<pre class='ls-progress-detail'>{html.escape(detail)}</pre>" if detail else ""
    return (
        f"<div class='ls-progress {html.escape(state)}' role='progressbar' aria-valuemin='0' aria-valuemax='100' "
        f"aria-valuenow='{value}' aria-label='{html.escape(label)}'>"
        f"<div class='ls-progress-head'><span>{html.escape(label)}</span>"
        f"<span class='ls-progress-pct'>{value}%</span></div>"
        f"<div class='ls-progress-track'><div class='ls-progress-fill' style='width:{value}%'></div></div>"
        f"{detail_html}</div>"
    )


STEP_LABELS = ("Pessoa", "Áudio", "Ajustes", "Geração", "Resultado")


def render_steps(completed: int = 0, current: Optional[int] = None) -> str:
    items = []
    for number, label in enumerate(STEP_LABELS, start=1):
        cls = " completed" if number <= completed else (" current" if number == current else "")
        items.append(f"<div class='ls-step{cls}'><span class='ls-step-dot'>{number}</span><span>{label}</span></div>")
    return "<div class='ls-stepper'>" + "".join(items) + "</div>"


def section_title(number: int, title: str, text: str, color: str = "") -> str:
    return (f"<div class='ls-section-title'><span class='ls-section-number {color}'>{number}</span>"
            f"<div><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p></div></div>")


def render_monitor(state: dict) -> str:
    """Cartão com o estado da tarefa em segundo plano (sobrevive ao F5)."""
    from lipsync_studio.media import format_time

    status = state.get("status", "idle")
    labels = {"idle": "Pronto para gerar", "running": "Gerando em segundo plano", "completed": "Concluído",
              "cancelled": "Cancelado", "error": "Falhou"}
    pct = float(state.get("pct") or 0)
    stage = html.escape(state.get("stage") or "")
    title = html.escape(state.get("title") or "")
    parts = [f"<div class='ls-monitor {status}'>",
             f"<div class='ls-monitor-head'><span class='ls-badge {status}'>{labels.get(status, status)}</span>"]
    if status != "idle":
        parts.append(f"<span class='ls-monitor-time'>⏱️ {format_time(state.get('elapsed', 0))}</span>")
    parts.append("</div>")
    if status == "idle":
        parts.append("<div class='ls-monitor-note'>Envie a pessoa e o áudio, ajuste e clique em <b>Gerar vídeo</b>. "
                     "A geração continua mesmo se você recarregar a página.</div>")
    else:
        if title:
            parts.append(f"<div class='ls-monitor-info'><b>{title}</b></div>")
        if stage:
            parts.append(f"<div class='ls-monitor-info'>{stage}</div>")
        if status == "running":
            parts.append(f"<div class='ls-bar'><span style='width:{max(2, min(100, pct)):.1f}%'></span></div>"
                         f"<div class='ls-monitor-note'>{pct:.0f}% · pode fechar ou recarregar a página; "
                         "o trabalho continua no Colab.</div>")
        if status == "error" and state.get("error"):
            parts.append(f"<div class='ls-problem'>{html.escape(str(state['error'])[:600])}</div>")
    parts.append("</div>")
    return "".join(parts)


def render_summary(session) -> str:
    """Resumo de uma sessão na aba de revisão."""
    from lipsync_studio.media import format_time
    from lipsync_studio.session import STATUS_LABELS

    if session is None:
        return "<div class='ls-empty'>Nenhuma sessão ainda. Gere um vídeo na primeira aba.</div>"
    counts = session.counts()
    chips = []
    for key, cls in (("done", "ok"), ("pending", "pending"), ("running", "pending"), ("error", "error")):
        n = counts.get(key, 0)
        chips.append(f"<span class='ls-chip {cls}{' zero' if not n else ''}'><b>{n}</b> "
                     f"{html.escape(STATUS_LABELS[key].split(' ', 1)[1])}</span>")
    s = session.settings
    return (
        "<div class='ls-summary'>"
        f"<div class='ls-summary-title'>{html.escape(session.title)}</div>"
        f"<div class='ls-meta'>Duração <b>{format_time(session.duration)}</b> · {len(session.parts)} parte(s) · "
        f"{s.resolution}p · {s.steps} passos · áudio {s.audio_guide:g} · semente {s.seed}"
        f"{' · TeaCache' if s.teacache else ''}</div>"
        f"<div class='ls-chips'>{''.join(chips)}</div></div>"
    )
