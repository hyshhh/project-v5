"""
logger.py - Pipeline 统一日志模块
=================================
原来散布在 pipeline.py / tracker.py / fps.py / detector.py 等文件里的
logging.info / logging.warning 全部用 pipe_log 替换。

输出格式:
  HH:MM:SS [pipeline]  LEVEL  event  key=value key=value ...

规则:
  - 每条日志一行，事件名精简，数据全部 key=value
  - 终端带色，文件无色
  - grep / awk / jq 友好
  - 无中文废话，无括号噪音，无重复摘要块

用法:
    from logger import pipe_log

    # track 生命周期
    pipe_log.enqueue(track=3, frame=270, queue=1)
    pipe_log.infer(track=3, frame=270)
    pipe_log.result(track=3, frame=270, string="21",
                    match="semantic", top3=['0123','0014','0789'])

    # FPS
    pipe_log.fps_stream(fps=10.2, frames=729, elapsed=66, tracks=3)
    pipe_log.fps_process(fps=10.2)

    # tracker
    pipe_log.cleanup(expired=[6])

    # pipeline 生命周期
    pipe_log.init(mode="concurrent", inference="agent", process_every=2)
    pipe_log.stats(frames=783, detections=42, tracks=7, recognized=3,
                   elapsed=71.0, fps=10.7, mode="concurrent")
    pipe_log.interrupted(frames=500, elapsed=45.2)

    # agent
    pipe_log.agent_init(method="langchain")
    pipe_log.agent_error(track=3, frame=270, error="timeout")
    pipe_log.agent_fallback(track=3, frame=270)
    pipe_log.trace_summary(count=9, entries=[...])

    # worker
    pipe_log.workers_start(n=4)
    pipe_log.workers_stop(remaining=2)

    # video
    pipe_log.video_open(source="rtsp://...", kind="stream")
    pipe_log.video_close(frames=783)

    # 通用
    pipe_log.info("custom_event", key1=val1, key2=val2)
    pipe_log.warn("slow_inference", track=3, ms=2500)
    pipe_log.error("yolo_crash", frame=100, err="CUDA OOM")
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional


# ── ANSI 颜色 ───────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    DIM    = "\033[2m"
    BOLD   = "\033[1m"
    INFO   = "\033[36m"      # 青
    WARN   = "\033[33m"      # 黄
    ERR    = "\033[31m"      # 红
    TRACK  = "\033[35m"      # 紫  track 事件
    FPS    = "\033[32m"      # 绿  fps
    AGENT  = "\033[33m"      # 黄  agent
    PIPE   = "\033[36m"      # 青  pipeline 生命周期
    VIDEO  = "\033[37m"      # 白  video
    WORKER = "\033[90m"      # 灰  worker
    MISC   = "\033[33m"      # 黄  自定义
    KV     = "\033[34m"      # 蓝  key
    TAG    = "\033[90m"      # 灰  [pipeline]


# ── 事件 → 颜色 ─────────────────────────────────────────
_EVENT_STYLE = {
    # track
    "enqueue":  C.TRACK,
    "infer":    C.TRACK,
    "result":   C.TRACK,
    "cleanup":  C.TRACK,
    # FPS
    "stream":   C.FPS,
    "process":  C.FPS,
    # pipeline
    "init":     C.PIPE,
    "stats":    C.PIPE,
    "interrupt":C.PIPE,
    # agent
    "agent_init":    C.AGENT,
    "agent_error":   C.AGENT,
    "agent_fallback":C.AGENT,
    "trace_summary": C.AGENT,
    # worker
    "workers_start": C.WORKER,
    "workers_stop":  C.WORKER,
    # video
    "video_open":    C.VIDEO,
    "video_close":   C.VIDEO,
}


# ── 带色格式化器 (终端) ─────────────────────────────────
class ColorFormatter(logging.Formatter):
    def format(self, record):
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        event = getattr(record, "ev", "msg")
        kv: dict = getattr(record, "kv", {})

        lc = {"INFO": C.INFO, "WARNING": C.WARN, "ERROR": C.ERR}.get(level, C.RESET)
        ec = _EVENT_STYLE.get(event, C.MISC)

        kvs = "  ".join(
            f"{C.KV}{k}{C.RESET}={C.DIM}{v}{C.RESET}" for k, v in kv.items()
        )
        return (
            f"{C.DIM}{ts}{C.RESET}  "
            f"{C.TAG}[pipeline]{C.RESET}  "
            f"{lc}{level:<7}{C.RESET}  "
            f"{ec}{event:<16}{C.RESET} {kvs}"
        )


# ── 无色格式化器 (文件) ─────────────────────────────────
class PlainFormatter(logging.Formatter):
    def format(self, record):
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        event = getattr(record, "ev", "msg")
        kv: dict = getattr(record, "kv", {})
        kvs = "  ".join(f"{k}={v}" for k, v in kv.items())
        return f"{ts}  [pipeline]  {level:<7}  {event:<16} {kvs}"


# ── 内部: 发出日志 ──────────────────────────────────────
def _emit(logger: logging.Logger, level: int, event: str, **kw):
    rec = logger.makeRecord(logger.name, level, "(pipe)", 0, "", (), None)
    rec.ev = event    # type: ignore
    rec.kv = kw       # type: ignore
    logger.handle(rec)


# ── 格式化值 ────────────────────────────────────────────
def _fmt(v: Any) -> Any:
    """把 list/dict 格式化得紧凑一点"""
    if isinstance(v, list) and len(v) > 5:
        return v[:5]
    return v


# ── PipelineLogger ──────────────────────────────────────
class PipelineLogger:
    """
    结构化日志，每条一行 key=value。
    替换掉所有分散的 logging.info("中文带括号 (%d)", n) 调用。
    """

    def __init__(self, name: str = "pipeline", level: int = logging.INFO):
        self._log = logging.getLogger(name)
        self._log.setLevel(level)
        if not self._log.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(ColorFormatter())
            self._log.addHandler(h)
            self._log.propagate = False

    # ══════════════════════════════════════════════════════
    #  Track 生命周期
    # ══════════════════════════════════════════════════════

    def enqueue(self, *, track: int, frame: int, queue: int):
        """新 track 入异步队列"""
        _emit(self._log, logging.INFO, "enqueue",
              track=track, frame=frame, queue=queue)

    def infer(self, *, track: int, frame: int):
        """异步推理开始"""
        _emit(self._log, logging.INFO, "infer",
              track=track, frame=frame)

    def result(self, *, track: int, frame: int,
               string: str, match: str,
               top3: Optional[List[str]] = None):
        """推理完成"""
        kv = dict(track=track, frame=frame, string=string, match=match)
        if top3:
            kv["top3"] = _fmt(top3)
        _emit(self._log, logging.INFO, "result", **kv)

    def cleanup(self, *, expired: List[int]):
        """清理过期 track"""
        _emit(self._log, logging.INFO, "cleanup",
              n=len(expired), tracks=_fmt(expired))

    def new_track(self, *, track: int, frame: int):
        """新 track 注册"""
        _emit(self._log, logging.INFO, "new_track",
              track=track, frame=frame)

    # ══════════════════════════════════════════════════════
    #  FPS
    # ══════════════════════════════════════════════════════

    def fps_stream(self, *, fps: float, frames: int, elapsed: int, tracks: int):
        """采集帧率"""
        _emit(self._log, logging.INFO, "stream",
              fps=f"{fps:.1f}", frames=frames,
              elapsed=f"{elapsed}s", tracks=tracks)

    def fps_process(self, *, fps: float):
        """处理帧率"""
        _emit(self._log, logging.INFO, "process", fps=f"{fps:.1f}")

    # ══════════════════════════════════════════════════════
    #  Pipeline 生命周期
    # ══════════════════════════════════════════════════════

    def init(self, *, mode: str, inference: str,
             process_every: int = 1,
             refresh: bool = False, refresh_gap: int = 0):
        """pipeline 初始化"""
        kv = dict(mode=mode, inference=inference, every=process_every)
        if refresh:
            kv["refresh"] = f"on(gap={refresh_gap})"
        _emit(self._log, logging.INFO, "init", **kv)

    def stats(self, *, frames: int, detections: int, tracks: int,
              recognized: int, elapsed: float, fps: float,
              mode: str, screenshots: int = 0):
        """处理完成统计 (替代原来 10 行 info)"""
        _emit(self._log, logging.INFO, "stats",
              frames=frames, detections=detections, tracks=tracks,
              recognized=recognized, elapsed=f"{elapsed:.1f}s",
              fps=f"{fps:.1f}", mode=mode, screenshots=screenshots)

    def interrupted(self, *, frames: int, elapsed: float):
        """用户中断"""
        _emit(self._log, logging.INFO, "interrupt",
              frames=frames, elapsed=f"{elapsed:.1f}s")

    # ══════════════════════════════════════════════════════
    #  Agent
    # ══════════════════════════════════════════════════════

    def agent_init(self, *, method: str):
        """Agent 初始化"""
        _emit(self._log, logging.INFO, "agent_init", method=method)

    def agent_error(self, *, track: int, frame: int, error: str):
        """Agent 推理出错"""
        _emit(self._log, logging.WARNING, "agent_error",
              track=track, frame=frame, error=error[:120])

    def agent_fallback(self, *, track: int, frame: int):
        """Agent 异常回退本地检索"""
        _emit(self._log, logging.WARNING, "agent_fallback",
              track=track, frame=frame)

    def agent_result(self, *, track: int, frame: int,
                     string: str, match: str,
                     candidates: Optional[List[str]] = None):
        """Agent 推理结果 (和 result 同结构，但标记来源)"""
        kv = dict(track=track, frame=frame, string=string, match=match)
        if candidates:
            kv["top3"] = _fmt(candidates)
        _emit(self._log, logging.INFO, "result", **kv)

    def trace_summary(self, *, count: int, entries: List[dict]):
        """Agent 运行链路摘要 (替代原来的 === 分隔块)"""
        for e in entries:
            _emit(self._log, logging.INFO, "trace",
                  type=e.get("type", ""),
                  track=e.get("track_id", -1),
                  frame=e.get("frame_id", -1),
                  msg=str(e.get("content", ""))[:80])

    # ══════════════════════════════════════════════════════
    #  Worker 线程
    # ══════════════════════════════════════════════════════

    def workers_start(self, *, n: int):
        """启动 worker 线程"""
        _emit(self._log, logging.INFO, "workers_start", n=n)

    def workers_stop(self, *, remaining: int = 0):
        """停止 worker"""
        kv = {}
        if remaining > 0:
            kv["remaining"] = remaining
        _emit(self._log, logging.INFO, "workers_stop", **kv)

    def worker_timeout(self, *, name: str):
        """worker 未在超时内退出"""
        _emit(self._log, logging.WARNING, "worker_timeout", name=name)

    # ══════════════════════════════════════════════════════
    #  Video
    # ══════════════════════════════════════════════════════

    def video_open(self, *, source: str, kind: str = "file"):
        """打开视频源"""
        # 截断长 URL
        src = source if len(source) < 60 else source[:57] + "..."
        _emit(self._log, logging.INFO, "video_open", source=src, kind=kind)

    def video_close(self, *, frames: int):
        """释放视频源"""
        _emit(self._log, logging.INFO, "video_close", frames=frames)

    def video_output(self, *, path: str):
        """输出视频写入"""
        _emit(self._log, logging.INFO, "video_output", path=path)

    def video_end(self):
        """视频源结束"""
        _emit(self._log, logging.INFO, "video_end")

    # ══════════════════════════════════════════════════════
    #  运行时控制
    # ══════════════════════════════════════════════════════

    def demo(self, *, enabled: bool):
        _emit(self._log, logging.INFO, "demo", enabled=enabled)

    def prompt_mode(self, *, mode: str):
        _emit(self._log, logging.INFO, "prompt_mode", mode=mode)

    def concurrent(self, *, enabled: bool):
        _emit(self._log, logging.INFO, "concurrent", enabled=enabled)

    # ══════════════════════════════════════════════════════
    #  通用 (自定义事件)
    # ══════════════════════════════════════════════════════

    def info(self, event: str, **kw: Any):
        _emit(self._log, logging.INFO, event, **kw)

    def warn(self, event: str, **kw: Any):
        _emit(self._log, logging.WARNING, event, **kw)

    def error(self, event: str, **kw: Any):
        _emit(self._log, logging.ERROR, event, **kw)

    # ══════════════════════════════════════════════════════
    #  文件输出 (无色)
    # ══════════════════════════════════════════════════════

    def add_file(self, path: str, level: int = logging.DEBUG):
        """追加文件日志 (无 ANSI 色)"""
        fh = logging.FileHandler(path)
        fh.setLevel(level)
        fh.setFormatter(PlainFormatter())
        self._log.addHandler(fh)


# ── 全局单例 ────────────────────────────────────────────
pipe_log = PipelineLogger()
