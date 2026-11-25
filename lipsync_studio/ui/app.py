"""Interface Gradio do LipSync Studio (InfiniteTalk).

    demo, launch_kwargs = build_app(studio, jobs, gpu_label="NVIDIA L4 · 24 GB")
    demo.queue().launch(share=True, **launch_kwargs)

Abas:
* 🎬 Dublar vídeo: pessoa (vídeo ou foto) + áudio -> vídeo com boca, rosto e corpo no ritmo da fala.
* 🧩 Revisão por partes: ouvir cada parte, gerar de novo com outra semente e remontar o vídeo final.
"""

from __future__ import annotations

import os
import random
from typing import Any, List, Optional

from lipsync_studio import __version__, media
from lipsync_studio.jobs import COMPLETED, RUNNING, JobManager
from lipsync_studio.media import format_time
from lipsync_studio.session import DEFAULT_PROMPT, STATUS_LABELS, Session, Settings
from lipsync_studio.studio import Studio, output_size
from lipsync_studio.ui.theme import (build_theme, load_css, render_engine_progress, render_header, render_monitor,
                                     render_steps, render_summary, section_title, style_kwargs)

KIND_VIDEO, KIND_IMAGE = "🎞️ Vídeo", "🖼️ Foto"
PRESETS = {"⚡ Rápido · 20 passos": 20, "⚖️ Equilibrado · 30 passos": 30, "💎 Qualidade · 40 passos (oficial)": 40}
DEFAULT_PRESET = "⚖️ Equilibrado · 30 passos"
RES_CHOICES = {"480p (recomendado)": "480", "720p (mais lento)": "720"}
AUDIO_LONGER = {"Vídeo vai e volta (recomendado)": "pingpong", "Vídeo recomeça do início": "loop",
                "Cortar o áudio no fim do vídeo": "trim"}
VIDEO_LONGER = {"Cortar o vídeo no fim do áudio": "trim", "Manter o vídeo inteiro (completa com silêncio)": "silence"}
PART_HEADERS = ["Parte", "Trecho", "Situação", "Semente", "Passos", "Tempo", "Observação"]

KEEPALIVE = """
<audio id="ls-keepalive" loop muted src="data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"></audio>
<script>
(function () {
  if (window.__lsSentry) return; window.__lsSentry = true;
  setInterval(function () { try { fetch(window.location.href, {method: 'HEAD', mode: 'no-cors'}).catch(function () {}); } catch (e) {} }, 25000);
  function unlock() { var a = document.getElementById('ls-keepalive'); if (a && a.paused) a.play().catch(function () {}); }
  document.addEventListener('pointerdown', unlock, {once: true});
  document.addEventListener('keydown', unlock, {once: true});
})();
</script>
"""


def _info_html(text: str, warn: bool = False) -> str:
    return f"<div class='ls-info{' warn' if warn else ''}'>{text}</div>" if text else ""


def describe_inputs(kind: str, video: Optional[str], image: Optional[str], audio: Optional[str],
                    resolution_label: str, audio_longer_label: str, video_longer_label: str) -> str:
    """Resumo do que foi enviado e do que vai acontecer (duração final, tamanho de saída)."""
    visual = image if kind == KIND_IMAGE else video
    lines, warn = [], False
    resolution = RES_CHOICES.get(resolution_label, "480")
    v_info = a_info = None
    try:
        if visual:
            v_info = media.probe(visual)
            if kind == KIND_IMAGE:
                lines.append(f"🖼️ Foto <b>{v_info.width}×{v_info.height}</b>")
            else:
                lines.append(f"🎞️ Vídeo <b>{v_info.describe_video()}</b>")
            ow, oh = output_size(v_info.width, v_info.height, resolution)
            lines.append(f"📐 Saída do InfiniteTalk em <b>{ow}×{oh}</b> (recorte central, 25 fps)")
        if audio:
            a_info = media.probe(audio)
            lines.append(f"🔊 Áudio <b>{format_time(a_info.duration)}</b>")
    except media.MediaError as exc:
        return _info_html(f"⚠️ {exc}", warn=True)
    if v_info and a_info and kind == KIND_VIDEO:
        diff = a_info.duration - v_info.duration
        if diff > 0.5:
            mode = AUDIO_LONGER.get(audio_longer_label, "pingpong")
            what = {"pingpong": "o vídeo vai e volta para cobrir o áudio",
                    "loop": "o vídeo recomeça do início para cobrir o áudio",
                    "trim": "o áudio é cortado no fim do vídeo"}[mode]
            lines.append(f"↔️ O áudio é {format_time(diff)} mais longo: {what}.")
        elif diff < -0.5:
            mode = VIDEO_LONGER.get(video_longer_label, "trim")
            what = {"trim": "o vídeo é cortado no fim do áudio",
                    "silence": "o vídeo fica inteiro e o áudio é completado com silêncio"}[mode]
            lines.append(f"↔️ O vídeo é {format_time(-diff)} mais longo: {what}.")
    if a_info and a_info.duration > 120:
        warn = True
        lines.append("⏳ Áudio longo: o InfiniteTalk é um modelo de 14B e leva vários minutos por parte. "
                     "Acompanhe o tempo estimado no painel de geração.")
    return _info_html("<br>".join(lines), warn)


def input_steps(kind: str, video: Optional[str], image: Optional[str], audio: Optional[str]) -> str:
    visual = image if kind == KIND_IMAGE else video
    if visual and audio:
        return render_steps(3, current=4)
    return render_steps(1 if visual else 0, current=2 if visual else 1)


def part_rows(session: Optional[Session]) -> List[List[Any]]:
    if session is None:
        return []
    return [[p.index, f"{format_time(p.start)} – {format_time(p.end)}", STATUS_LABELS.get(p.status, p.status),
             p.seed if p.seed is not None else "", p.steps or "", format_time(p.seconds) if p.seconds else "",
             p.message] for p in session.parts]


def build_app(studio: Studio, jobs: JobManager, gpu_label: Optional[str] = None, gr=None):
    """Monta a interface. Devolve (demo, kwargs do launch)."""
    if gr is None:
        import gradio as gr

    blocks_style, launch_style = style_kwargs(gr, build_theme(gr), load_css())
    engine = studio.engine

    def engine_state() -> dict:
        status = engine.status() if hasattr(engine, "status") else {"state": "ready", "detail": ""}
        if status.get("state") == "ready" and jobs.is_running():
            status = {**status, "state": "busy"}
        return status

    def engine_bar(status: dict) -> str:
        state = status.get("state")
        if state in ("installing", "loading"):
            label = "Instalando o InfiniteTalk..." if state == "installing" else "Carregando o InfiniteTalk..."
            return render_engine_progress(60 if state == "loading" else 30, label, status.get("detail", ""))
        if state == "error" and status.get("detail"):
            return render_engine_progress(100, "O InfiniteTalk precisa ser reiniciado", status["detail"], "error")
        return render_engine_progress(None)

    # ------------------------------------------------------------------ ações da aba principal
    def toggle_kind(kind):
        is_image = kind == KIND_IMAGE
        return (gr.update(visible=not is_image), gr.update(visible=is_image), gr.update(visible=not is_image),
                gr.update(visible=not is_image))

    def apply_preset(label):
        return gr.update(value=PRESETS.get(label, 30))

    def start_generation(kind, video, image, audio, prompt, res_label, steps, audio_guide, text_guide, seed,
                         teacache, part_seconds, audio_longer_label, video_longer_label, bg_volume):
        visual = image if kind == KIND_IMAGE else video
        if not visual:
            return "⚠️ Envie o " + ("a foto" if kind == KIND_IMAGE else "vídeo") + " da pessoa."
        if not audio:
            return "⚠️ Envie o áudio com a fala."
        settings = Settings(prompt=prompt, resolution=RES_CHOICES.get(res_label, "480"), steps=int(steps),
                            audio_guide=float(audio_guide), text_guide=float(text_guide),
                            seed=int(seed if seed is not None else 42), teacache=bool(teacache),
                            part_seconds=int(part_seconds),
                            audio_longer=AUDIO_LONGER.get(audio_longer_label, "pingpong"),
                            video_longer=VIDEO_LONGER.get(video_longer_label, "trim"),
                            background_volume=float(bg_volume or 0)).validate()
        title = f"Dublagem de {os.path.basename(visual)}"
        ok, msg = jobs.start("run", title, lambda ctx: studio.run(visual, audio, settings, ctx))
        return "🚀 Geração iniciada. Acompanhe abaixo." if ok else f"⚠️ {msg}"

    def stop_job():
        return jobs.stop()

    def history():
        return studio.outputs() or None

    # ------------------------------------------------------------------ ações da revisão
    def session_choices():
        return [(s.display_name(), s.id) for s in studio.sessions()]

    def refresh_sessions(current):
        choices = session_choices()
        ids = [c[1] for c in choices]
        value = current if current in ids else (ids[0] if ids else None)
        return gr.update(choices=choices, value=value)

    def load_session(sid) -> Optional[Session]:
        if not sid:
            return None
        try:
            return studio.load(sid)
        except (OSError, ValueError, KeyError):
            return None

    def show_session(sid):
        session = load_session(sid)
        choices = [(p.label(), p.index) for p in session.parts] if session else []
        return (render_summary(session), part_rows(session),
                gr.update(choices=choices, value=choices[0][1] if choices else None))

    def preview_path(session: Session, part, kind: str) -> Optional[str]:
        src = session.part_input_video(part) if kind == "in" else session.part_output(part)
        if not os.path.exists(src):
            return None
        folder = session.path("previews")
        os.makedirs(folder, exist_ok=True)
        dst = os.path.join(folder, f"{part.name}_{kind}.mp4")
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            media.preview_with_audio(src, session.part_input_audio(part), dst)
        return dst

    def show_part(sid, index):
        session = load_session(sid)
        if session is None or index in (None, ""):
            return None, None, "", gr.update()
        try:
            part = session.get_part(int(index))
            before, after = preview_path(session, part, "in"), preview_path(session, part, "out")
        except (KeyError, media.MediaError) as exc:
            return None, None, f"⚠️ {exc}", gr.update()
        info = (f"**{part.label()}** · {STATUS_LABELS.get(part.status, part.status)}"
                + (f" · semente {part.seed}" if part.seed is not None else "")
                + (f" · {part.message}" if part.message else ""))
        seed = part.seed if part.seed is not None else session.settings.seed
        return before, after, info, gr.update(value=seed)

    def random_seed():
        return random.randint(0, 99_999_999)

    def regenerate_part(sid, index, seed, steps, audio_guide):
        session = load_session(sid)
        if session is None or index in (None, ""):
            return "⚠️ Escolha a sessão e a parte."
        idx = int(index)
        ok, msg = jobs.start("part", f"Parte {idx} de {session.title}",
                             lambda ctx: studio.generate(studio.load(sid), ctx, [idx], int(seed), int(steps),
                                                         float(audio_guide)))
        return "🔁 Gerando a parte de novo. Depois clique em **Montar vídeo final**." if ok else f"⚠️ {msg}"

    def run_pending(sid):
        session = load_session(sid)
        if session is None:
            return "⚠️ Escolha uma sessão."

        def work(ctx):
            current = studio.load(sid)
            failed = studio.generate(current, ctx)
            if not failed and current.ready_to_assemble():
                return {"session": sid, "output": studio.assemble(current, ctx)}
            return {"session": sid, "output": None}

        ok, msg = jobs.start("pending", f"Partes pendentes de {session.title}", work)
        return "▶️ Gerando as partes pendentes." if ok else f"⚠️ {msg}"

    def assemble(sid):
        session = load_session(sid)
        if session is None:
            return "⚠️ Escolha uma sessão."
        ok, msg = jobs.start("assemble", f"Montagem de {session.title}",
                             lambda ctx: {"session": sid, "output": studio.assemble(studio.load(sid), ctx)})
        return "🧩 Montando o vídeo final." if ok else f"⚠️ {msg}"

    # ------------------------------------------------------------------ sincronização periódica (F5-safe)
    def tick(seen, review_sid):
        st = jobs.state()
        status = engine_state()
        out = [render_monitor(st), st["logs"], render_header(gpu_label, status["state"], __version__),
               engine_bar(status)]
        skip = gr.skip()
        finished = st["status"] != RUNNING and st["job_id"] and st["job_id"] != seen
        if st["status"] == RUNNING:
            out.append(render_steps(3, current=4) if st["kind"] == "run" else skip)
        elif finished and st["status"] == COMPLETED and st["kind"] == "run":
            out.append(render_steps(5))
        else:
            out.append(skip)
        if not finished:
            return out + [skip, skip, skip, skip, skip, skip, seen]
        result = st.get("result") if st["status"] == COMPLETED else None
        output = result.get("output") if isinstance(result, dict) else None
        video = output if output and os.path.exists(output) else skip
        if review_sid:
            review_summary, review_rows, _ = show_session(review_sid)
        else:
            review_summary = review_rows = skip
        return out + [video, output if output else skip, history(), video, review_summary, review_rows,
                      st["job_id"]]

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title="LipSync Studio · InfiniteTalk", **blocks_style) as demo:
        gr.HTML(KEEPALIVE)
        engine_progress = gr.HTML(render_engine_progress(None), elem_id="engine-progress-slot")
        header = gr.HTML(render_header(gpu_label, engine_state()["state"], __version__))
        seen_job = gr.State(jobs.state()["job_id"])

        with gr.Tabs(elem_id="main-tabs"):
            # ========================================================== aba 1
            with gr.TabItem("🎬 Dublar vídeo"):
                steps_html = gr.HTML(render_steps(0, current=1))
                with gr.Column(elem_id="dub-workspace"):
                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(1, "Pessoa", "Envie um vídeo da pessoa falando ou uma foto do rosto."))
                        kind = gr.Radio([KIND_VIDEO, KIND_IMAGE], value=KIND_VIDEO, label="Tipo de entrada",
                                        info="Vídeo: o InfiniteTalk segue os movimentos originais. Foto: anima a pessoa do zero.")
                        video_in = gr.Video(label="Vídeo (MP4, MOV, MKV, WEBM)", sources=["upload"])
                        image_in = gr.Image(label="Foto (PNG ou JPG)", type="filepath", sources=["upload"],
                                            visible=False)
                        gr.HTML("<p class='ls-hint'>Funciona melhor com <b>uma pessoa</b>, rosto visível e bem "
                                "iluminado. O modelo recria o quadro inteiro em 480p ou 720p.</p>")

                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(2, "Áudio da fala", "A voz que a pessoa vai falar (ex.: a dublagem "
                                              "gerada no Matraca Studio).", "blue"))
                        audio_in = gr.Audio(label="Áudio (WAV, MP3, M4A, OGG)", type="filepath",
                                            sources=["upload", "microphone"])
                        inputs_info = gr.HTML("")

                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(3, "Ajustes", "Descreva a cena e escolha velocidade e resolução.",
                                              "violet"))
                        prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Descrição da cena (prompt)", lines=2,
                                            info="Em inglês funciona melhor. Ex.: 'A woman is talking in a studio'.")
                        with gr.Row():
                            res = gr.Radio(list(RES_CHOICES), value="480p (recomendado)", label="Resolução")
                            preset = gr.Radio(list(PRESETS), value=DEFAULT_PRESET, label="Velocidade × qualidade")
                        with gr.Accordion("⚙️ Ajustes avançados", open=False):
                            with gr.Row():
                                steps = gr.Slider(4, 50, value=PRESETS[DEFAULT_PRESET], step=1,
                                                  label="Passos de amostragem")
                                audio_guide = gr.Slider(1, 8, value=4.0, step=0.5, label="Força do áudio",
                                                        info="Mais alto = boca mais fiel (3 a 5 é o ideal).")
                                text_guide = gr.Slider(1, 8, value=5.0, step=0.5, label="Força do texto")
                            with gr.Row():
                                seed = gr.Number(value=42, precision=0, label="Semente")
                                teacache = gr.Checkbox(value=True, label="TeaCache (≈2× mais rápido)")
                                part_seconds = gr.Slider(10, 90, value=30, step=5, label="Tamanho de cada parte (s)",
                                                         info="Partes menores usam menos memória e podem ser refeitas.")
                            with gr.Row():
                                audio_longer = gr.Radio(list(AUDIO_LONGER), value="Vídeo vai e volta (recomendado)",
                                                        label="Se o áudio for mais longo que o vídeo")
                                video_longer = gr.Radio(list(VIDEO_LONGER), value="Cortar o vídeo no fim do áudio",
                                                        label="Se o vídeo for mais longo que o áudio")
                            bg_volume = gr.Slider(0, 1, value=0, step=0.05,
                                                  label="Som original do vídeo ao fundo (0 = desligado)",
                                                  info="Útil para manter música ou ambiente; a voz original também volta.")

                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(4, "Geração", "Roda em segundo plano no Colab: pode recarregar a página."))
                        with gr.Row():
                            generate_btn = gr.Button("▶ Gerar vídeo", variant="primary", scale=3,
                                                     elem_id="generate-button")
                            stop_btn = gr.Button("■ Cancelar", variant="stop", scale=1, elem_id="stop-button")
                        action_msg = gr.Markdown("")
                        monitor = gr.HTML(render_monitor(jobs.state()))
                        with gr.Accordion("🖥️ Registros", open=False):
                            logs = gr.Textbox(lines=14, max_lines=14, show_label=False, elem_id="console-logs",
                                              interactive=False, autoscroll=True)

                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(5, "Resultado", "Vídeo final com o áudio original em alta qualidade."))
                        result_video = gr.Video(label="Vídeo dublado", interactive=False)
                        result_file = gr.File(label="Baixar MP4", interactive=False)
                        with gr.Accordion("📁 Vídeos desta sessão do Colab", open=False):
                            history_files = gr.Files(value=history(), label="Gerados", interactive=False)

            # ========================================================== aba 2
            with gr.TabItem("🧩 Revisão por partes"):
                with gr.Column(elem_id="review-workspace"):
                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(1, "Sessão", "Cada geração fica salva em partes em sessoes/."))
                        with gr.Row():
                            session_dd = gr.Dropdown(choices=session_choices(), label="Sessão", scale=4)
                            refresh_btn = gr.Button("🔄 Atualizar", scale=1)
                        summary = gr.HTML(render_summary(None))
                        table = gr.Dataframe(headers=PART_HEADERS, value=[], interactive=False, wrap=True,
                                             elem_id="parts-table")

                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(2, "Ouvir e refazer uma parte",
                                              "Compare o original com o gerado. Se não gostar, gere de novo com outra semente.",
                                              "blue"))
                        part_dd = gr.Dropdown(choices=[], label="Parte")
                        part_info = gr.Markdown("")
                        with gr.Row():
                            part_before = gr.Video(label="Original (com o novo áudio)", interactive=False)
                            part_after = gr.Video(label="InfiniteTalk", interactive=False)
                        with gr.Row():
                            regen_seed = gr.Number(value=42, precision=0, label="Semente", scale=2)
                            dice_btn = gr.Button("🎲 Sortear", scale=1)
                            regen_steps = gr.Slider(4, 50, value=30, step=1, label="Passos", scale=3)
                            regen_audio = gr.Slider(1, 8, value=4.0, step=0.5, label="Força do áudio", scale=3)
                        regen_btn = gr.Button("🔁 Gerar esta parte de novo", elem_id="regen-button")

                    with gr.Column(elem_classes=["ls-card"]):
                        gr.HTML(section_title(3, "Remontar", "Junte as partes no vídeo final.", "violet"))
                        with gr.Row():
                            pending_btn = gr.Button("▶ Gerar partes pendentes", elem_id="pending-button", scale=2)
                            assemble_btn = gr.Button("🧩 Montar vídeo final", elem_id="assemble-button", scale=2)
                            stop_review_btn = gr.Button("■ Cancelar", variant="stop", elem_id="stop-review-button",
                                                        scale=1)
                        review_msg = gr.Markdown("")
                        review_video = gr.Video(label="Vídeo final", interactive=False)

        # ------------------------------------------------------------------ eventos
        kind.change(toggle_kind, [kind], [video_in, image_in, video_longer, bg_volume], queue=False)
        preset.change(apply_preset, [preset], [steps], queue=False)
        info_inputs = [kind, video_in, image_in, audio_in, res, audio_longer, video_longer]
        for comp in (kind, video_in, image_in, audio_in, res, audio_longer, video_longer):
            comp.change(describe_inputs, info_inputs, [inputs_info], show_progress="hidden")
        for comp in (kind, video_in, image_in, audio_in):
            comp.change(input_steps, [kind, video_in, image_in, audio_in], [steps_html], queue=False,
                        show_progress="hidden")

        generate_btn.click(start_generation,
                           [kind, video_in, image_in, audio_in, prompt, res, steps, audio_guide, text_guide, seed,
                            teacache, part_seconds, audio_longer, video_longer, bg_volume],
                           [action_msg], show_progress="hidden")
        stop_btn.click(stop_job, None, [action_msg], queue=False, show_progress="hidden")
        stop_review_btn.click(stop_job, None, [review_msg], queue=False, show_progress="hidden")

        refresh_btn.click(refresh_sessions, [session_dd], [session_dd], queue=False)
        session_dd.change(show_session, [session_dd], [summary, table, part_dd], show_progress="hidden")
        part_dd.change(show_part, [session_dd, part_dd], [part_before, part_after, part_info, regen_seed],
                       show_progress="minimal")
        dice_btn.click(random_seed, None, [regen_seed], queue=False)
        regen_btn.click(regenerate_part, [session_dd, part_dd, regen_seed, regen_steps, regen_audio], [review_msg],
                        show_progress="hidden")
        pending_btn.click(run_pending, [session_dd], [review_msg], show_progress="hidden")
        assemble_btn.click(assemble, [session_dd], [review_msg], show_progress="hidden")

        timer = gr.Timer(1.0)
        timer.tick(tick, [seen_job, session_dd],
                   [monitor, logs, header, engine_progress, steps_html,
                    result_video, result_file, history_files, review_video, summary, table, seen_job],
                   queue=False, show_progress="hidden")
        demo.load(refresh_sessions, [session_dd], [session_dd], queue=False)

    return demo, launch_style


def launch(demo, studio: Studio, launch_style: dict, share: bool = True, **kwargs):
    """Fila com no máximo 32 pedidos; a geração em si roda na fila de tarefas (uma por vez)."""
    return demo.queue(max_size=32, default_concurrency_limit=4).launch(
        share=share, allowed_paths=[studio.outputs_dir, studio.sessions_dir], **launch_style, **kwargs)
