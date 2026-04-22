"""
logger.py - Pipeline 统一日志模块
=================================
将原来散乱的日志统一为：
  HH:MM:SS [pipeline]  LEVEL  event  k=v k=v ...

事件名精简，数据用 key=value，终端彩色，文件无色。

用法:
    from logger import pipe_log

    pipe_log.enqueue(track=3, frame=270, queue=1)
    pipe_log.infer(track=3, frame=270)
    pipe_log.result(track=3, frame=270, string="21",
                    match="semantic", top3=['0123','0014','0789'])
    pipe_log.fps_stream(fps=10.2, frames=729, elapsed=66, tracks=3)
    pipe_log.fps_process(fps=10.2)
    pipe_log.cleanup(expired=[6])
"""

import logging
import sys
import time
from typing import List, Optional


# ── 颜色 ─────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    DIM    = "\033[2m"
    BOLD   = "\033[1m"
    INFO   = "\033[36m"      # 青
    WARN   = "\033[33m"      # 黄
    ERR    = "\033[31m"      # 红
    TRACK  = "\033[35m"      # 紫  track 事件
    FPS    = "\033[32m"      # 绿  fps 事件
    MISC   = "\033[33m"      # 黄  其他事件
    KV     = "\033[34m"      # 蓝  key
    TAG    = "\033[90m"      # 灰  [module]


# ── 事件 → 颜色映射 ─────────────────────────────────────
_EVENT_STYLE = {
    "enqueue": C.TRACK,
    "infer":   C.TRACK,
    "result":  C.TRACK,
    "stream":  C.FPS,
    "process": C.FPS,
}


# ── 带色格式化器 (终端) ─────────────────────────────────
class ColorFormatter(logging.Formatter):
    def format(self, record):
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        event = getattr(record, "ev", "msg")
        kv: dict = getattr(record, "kv", {})

        # 颜色
        lc = {"INFO": C.INFO, "WARNING": C.WARN, "ERROR": C.ERR}.get(level, C.RESET)
        ec = _EVENT_STYLE.get(event, C.MISC)

        # key=value 拼接
        kvs = "  ".join(
            f"{C.KV}{k}{C.RESET}={C.DIM}{v}{C.RESET}" for k, v in kv.items()
        )

        return f"{C.DIM}{ts}{C.RESET}  {C.TAG}[pipeline]{C.RESET}  {lc}{level:<7}{C.RESET}  {ec}{event:<8}{C.RESET} {kvs}"


# ── 无色格式化器 (文件) ─────────────────────────────────
class PlainFormatter(logging.Formatter):
    def format(self, record):
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        event = getattr(record, "ev", "msg")
        kv: dict = getattr(record, "kv", {})
        kvs = "  ".join(f"{k}={v}" for k, v in kv.items())
        return f"{ts}  [pipeline]  {level:<7}  {event:<8} {kvs}"


# ── 内部: 发出一条日志 ──────────────────────────────────
def _log(logger: logging.Logger, level: int, event: str, **kw):
    rec = logger.makeRecord(logger.name, level, "(pipe)", 0, "", (), None)
    rec.ev = event    # type: ignore
    rec.kv = kw       # type: ignore
    logger.handle(rec)


# ── PipelineLogger 公开接口 ─────────────────────────────
class PipelineLogger:
    """
    每个 track 操作一行，关键信息全部 key=value，
    grep / awk / jq 友好，终端一眼扫完。
    """

    def __init__(self, name: str = "pipeline", level: int = logging.INFO):
        self._log = logging.getLogger(name)
        self._log.setLevel(level)
        if not self._log.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(ColorFormatter())
            self._log.addHandler(h)
            self._log.propagate = False

    # ── track 生命周期 ────────────────────────────────────

    def enqueue(self, *, track: int, frame: int, queue: int):
        """新 track 入队"""
        _log(self._log, logging.INFO, "enqueue",
             track=track, frame=frame, queue=queue)

    def infer(self, *, track: int, frame: int):
        """异步推理开始"""
        _log(self._log, logging.INFO, "infer",
             track=track, frame=frame)

    def result(self, *, track: int, frame: int,
               string: str, match: str,
               top3: Optional[List[str]] = None):
        """推理完成"""
        kv = dict(track=track, frame=frame, string=string, match=match)
        if top3:
            kv["top3"] = top3
        _log(self._log, logging.INFO, "result", **kv)

    # ── FPS ──────────────────────────────────────────────

    def fps_stream(self, *, fps: float, frames: int, elapsed: int, tracks: int):
        """采集帧率"""
        _log(self._log, logging.INFO, "stream",
             fps=fps, frames=frames, elapsed=f"{elapsed}s", tracks=tracks)

    def fps_process(self, *, fps: float):
        """处理帧率"""
        _log(self._log, logging.INFO, "process", fps=fps)

    # ── tracker 管理 ─────────────────────────────────────

    def cleanup(self, *, expired: List[int]):
        """清理过期 track"""
        _log(self._log, logging.INFO, "cleanup",
             n=len(expired), tracks=expired)

    # ── 通用 ─────────────────────────────────────────────

    def info(self, event: str, **kw):
        _log(self._log, logging.INFO, event, **kw)

    def warn(self, event: str, **kw):
        _log(self._log, logging.WARNING, event, **kw)

    def error(self, event: str, **kw):
        _log(self._log, logging.ERROR, event, **kw)

    # ── 添加文件输出 (无色) ──────────────────────────────

    def add_file(self, path: str, level: int = logging.DEBUG):
        fh = logging.FileHandler(path)
        fh.setLevel(level)
        fh.setFormatter(PlainFormatter())
        self._log.addHandler(fh)


# ── 全局单例 ────────────────────────────────────────────
pipe_log = PipelineLogger()
