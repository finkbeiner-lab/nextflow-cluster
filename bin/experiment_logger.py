"""Experiment-scoped debug log shared across every worker in a pipeline run.

Parallel Slurm jobs append to a single ``<output_path>/pipeline_debug.log``
under ``fcntl.flock`` — same pattern as ``tracking_montage._append_df_locked``.
The existing ``./finkbeiner_logs/<module>.log`` handlers are left alone; this
is an ADDITIONAL sink attached to the same logger.
"""
from __future__ import annotations
import fcntl, logging, os, sys
from typing import Optional

LOG_FILENAME = "pipeline_debug.log"
_FMT = "%(asctime)s.%(msecs)03d [%(module_tag)s] %(levelname)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _FlockAppendHandler(logging.Handler):
    """flock-locks the file around every append; falls back to lockf then
    unlocked O_APPEND (short-line writes stay kernel-atomic)."""
    def __init__(self, path: str) -> None:
        super().__init__(); self._path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            with open(self._path, "a", encoding="utf-8") as handle:
                fd, locker = handle.fileno(), None
                for candidate in (fcntl.flock, fcntl.lockf):
                    try:
                        candidate(fd, fcntl.LOCK_EX); locker = candidate; break
                    except OSError: continue
                try:
                    handle.write(line); handle.flush()
                finally:
                    if locker is not None:
                        try: locker(fd, fcntl.LOCK_UN)
                        except OSError: pass
        except Exception:  # noqa: BLE001 - never propagate from a log handler
            self.handleError(record)


class _TagFilter(logging.Filter):
    """Injects ``module_tag`` onto every record for the formatter."""
    def __init__(self, module_name: str) -> None:
        super().__init__(); self._tag = module_name
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "module_tag"): record.module_tag = self._tag
        return True


def _resolve_dir(output_path: Optional[str]) -> str:
    """Explicit arg -> ``NEXTFLOW_OUTPUT_PATH`` env -> ``os.getcwd()``."""
    for candidate in (output_path, os.environ.get("NEXTFLOW_OUTPUT_PATH", "")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return os.getcwd()


def attach_experiment_log(
    logger: logging.Logger, output_path: str, module_name: str
) -> None:
    """Attach a flock-protected FileHandler on ``<output_path>/pipeline_debug.log``.

    Idempotent per (logger, path). Failures never raise — the pipeline
    must keep running even if the debug log cannot be attached.
    """
    try:
        target_dir = _resolve_dir(output_path)
        os.makedirs(target_dir, exist_ok=True)
        log_path = os.path.join(target_dir, LOG_FILENAME)
        marker = f"_exp_log_attached::{log_path}"
        if getattr(logger, marker, False): return
        handler = _FlockAppendHandler(log_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        handler.addFilter(_TagFilter(module_name))
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        logger.addHandler(handler); setattr(logger, marker, True)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[experiment_logger] failed to attach for {module_name}: {exc}\n"
        )
