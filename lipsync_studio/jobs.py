"""Fila de uma tarefa por vez, em segundo plano.

A geração continua mesmo se a página for recarregada (F5) ou a conexão oscilar: a interface só lê o
estado daqui a cada segundo e recupera etapa, progresso, registros e resultado.
"""

from __future__ import annotations

import collections
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

from lipsync_studio.engine import Cancelled
from lipsync_studio.studio import Context

IDLE, RUNNING, COMPLETED, CANCELLED, ERROR = "idle", "running", "completed", "cancelled", "error"


class JobContext(Context):
    def __init__(self, manager: "JobManager"):
        self.manager = manager

    def log(self, message: str) -> None:
        self.manager.log(message)

    def stage(self, text: str, pct: Optional[float] = None) -> None:
        self.manager._update(stage=text, pct=pct)

    def cancelled(self) -> bool:
        return self.manager._cancel.is_set()


class JobManager:
    def __init__(self, echo: bool = True, max_lines: int = 800):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logs: collections.deque = collections.deque(maxlen=max_lines)
        self.echo = echo
        self._state: Dict[str, Any] = {"job_id": None, "kind": "", "title": "", "status": IDLE, "stage": "",
                                       "pct": 0.0, "started": None, "finished": None, "result": None,
                                       "error": ""}

    # ------------------------------------------------------------------ estado
    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            for line in str(message).splitlines() or [""]:
                self._logs.append(f"[{stamp}] {line}")
        if self.echo:
            print(message, flush=True)

    def _update(self, **items) -> None:
        with self._lock:
            if items.get("pct") is None:
                items.pop("pct", None)
            self._state.update(items)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def state(self) -> Dict[str, Any]:
        with self._lock:
            st = dict(self._state)
            st["logs"] = "\n".join(self._logs)
        end = st["finished"] or time.time()
        st["elapsed"] = (end - st["started"]) if st["started"] else 0.0
        return st

    # ------------------------------------------------------------------ controle
    def start(self, kind: str, title: str, fn: Callable[..., Any], *args, **kwargs) -> tuple:
        """Inicia ``fn(ctx, *args, **kwargs)`` em segundo plano. Devolve (ok, mensagem)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, f"Já existe uma tarefa em andamento: {self._state['title']}."
            self._cancel.clear()
            self._logs.clear()
            job_id = uuid.uuid4().hex[:8]
            self._state.update(job_id=job_id, kind=kind, title=title, status=RUNNING, stage="Iniciando...",
                               pct=0.0, started=time.time(), finished=None, result=None, error="")
            ctx = JobContext(self)
            self._thread = threading.Thread(target=self._run, args=(ctx, fn, args, kwargs), daemon=True,
                                            name=f"job-{kind}-{job_id}")
            self._thread.start()
        return True, job_id

    def _run(self, ctx: JobContext, fn, args, kwargs) -> None:
        try:
            result = fn(ctx, *args, **kwargs)
            self._update(status=COMPLETED, result=result, pct=100.0, stage="Concluído", finished=time.time())
        except Cancelled:
            self.log("⏹️ Cancelado.")
            self._update(status=CANCELLED, stage="Cancelado", finished=time.time())
        except BaseException as exc:
            self.log(f"❌ {type(exc).__name__}: {exc}")
            self.log(traceback.format_exc()[-2500:])
            self._update(status=ERROR, error=str(exc) or type(exc).__name__, stage="Falhou", finished=time.time())

    def stop(self) -> str:
        if not self.is_running():
            return "Nenhuma tarefa em andamento."
        self._cancel.set()
        self.log("⏹️ Cancelamento pedido: a geração para no próximo passo.")
        return "Cancelando... a geração para no próximo passo."

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.state()
