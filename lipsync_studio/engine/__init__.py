"""Motor InfiniteTalk (MeiGen-AI) num venv próprio, com worker HTTP local.

Este arquivo só usa a biblioteca padrão: ele é importado tanto no notebook quanto dentro do venv.
"""


class EngineError(RuntimeError):
    """Falha ao instalar, iniciar ou usar o modelo."""


class OutOfMemoryError(EngineError):
    """A GPU ficou sem memória (parte longa demais ou resolução alta demais)."""


class Cancelled(Exception):
    """O usuário cancelou o processamento."""
