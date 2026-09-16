# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json, os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Any, Dict
# === 全局错误/成本统计与熔断控制 ===

STOP_ALL_FILE = Path("STOP_ALL")
STOP_REASON_FILE = Path("STOP_ALL_REASON.txt")

# 允许你自定义触发熔断的阈值（根据实际计费情况调整）
ERROR_THRESHOLD = 10          # 一定时间内允许的最大错误数
ERROR_WINDOW_S = 300          # 统计错误数的时间窗口，单位秒
COST_THRESHOLD = 50.0         # 累积费用上限（美元为例）

_error_timestamps = []
_cost_total = 0.0


# 统一的状态目录
_STATUS_DIR = Path("results/.runtime")


CIRCUIT_FILE = Path(".circuit_state.json")  # 进程/目录级的熔断状态
KILL_SWITCH = Path("STOP_ALL")              # 全局紧急止损开关（创建该文件即停）
DEFAULT_TIMEOUT = 90                        # 单次请求硬超时（秒）
DEFAULT_BACKOFF_MAX = 30                    # 指数退避的单次最大等待（秒）
DEFAULT_WAIT_BUDGET = 180                   # 单窗口/单请求族的累计等待上限（秒）
DEFAULT_NET_ERR_WINDOW = 60                 # 熔断统计窗口（秒）
DEFAULT_NET_ERR_THRESHOLD = 3               # 熔断触发阈值（窗口内错误次数）
DEFAULT_COOLDOWN = 90                       # 熔断冷却期（秒）

def _now() -> float: return time.time()

def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def circuit_is_open() -> bool:
    if KILL_SWITCH.exists():   # 紧急止损：立即停
        return True
    st = _read_json(CIRCUIT_FILE)
    if not st: return False
    if st.get("state") != "OPEN": return False
    if _now() >= st.get("until", 0):
        # 进入 Half-Open，允许做一次探针
        st["state"] = "HALF"
        _write_json(CIRCUIT_FILE, st)
        return False
    return True

def circuit_open(cooldown: int = DEFAULT_COOLDOWN, reason: str = "network"):
    st = dict(state="OPEN", until=_now() + cooldown, reason=reason, ts=_now())
    _write_json(CIRCUIT_FILE, st)

def circuit_close():
    _write_json(CIRCUIT_FILE, dict(state="CLOSED", ts=_now()))

def record_net_error():
    st = _read_json(CIRCUIT_FILE)
    hist = st.get("hist", [])
    hist = [x for x in hist if _now() - x < DEFAULT_NET_ERR_WINDOW]
    hist.append(_now())
    st["hist"] = hist
    if len(hist) >= DEFAULT_NET_ERR_THRESHOLD:
        circuit_open()
    else:
        _write_json(CIRCUIT_FILE, st)

def half_open_probe_ok():
    st = _read_json(CIRCUIT_FILE)
    if st.get("state") == "HALF":
        circuit_close()

def guarded_request(
    fn: Callable[[], Any],
    max_retries: int = 5,
    base_delay: int = 2,
    timeout: int = DEFAULT_TIMEOUT,
    wait_budget: int = DEFAULT_WAIT_BUDGET,
    on_log: Optional[Callable[[str], None]] = None,
) -> Any:
    """
    对单次“请求族”（一次业务调用）做带超时、指数退避（上限）、累计等待上限、熔断联动的重试。
    """
    log = (lambda s: None) if on_log is None else on_log
    total_wait = 0
    for i in range(max_retries):
        if circuit_is_open():
            raise RuntimeError("CircuitOpen: 网络异常冷却中或紧急止损开启")
        try:
            # 执行请求并捕获返回值，便于做费用统计（若 SDK/返回结构提供 usage）
            resp = fn()

            # —— 费用统计（若响应包含 usage / total_cost 字段） ——
            try:
                cost = 0.0
                # 响应可能是 dict（如 web.run style）或对象（如 SDK）
                if isinstance(resp, dict):
                    usage = resp.get("usage") or {}
                    cost = float(usage.get("total_cost", 0.0) or 0.0)
                else:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        # 支持 usage.total_cost 属性或 usage.get(...)
                        cost = float(getattr(usage, "total_cost",
                                             getattr(usage, "get", lambda k, d=None: d)("total_cost", 0.0)) or 0.0)
            except Exception:
                cost = 0.0

            # 累加成本并判定阈值
            global _cost_total
            _cost_total += float(cost or 0.0)
            if _cost_total >= COST_THRESHOLD:
                try:
                    STOP_ALL_FILE.write_text("cost_threshold_reached\n", encoding="utf-8")
                    STOP_REASON_FILE.write_text(f"[熔断] total cost {_cost_total:.2f} >= {COST_THRESHOLD}\n",
                                                encoding="utf-8")
                except Exception:
                    pass
                print("🚨 触发 STOP_ALL：累计费用超过阈值", flush=True)
                raise RuntimeError("Global STOP_ALL triggered (cost limit)")

            # 成功返回
            return resp

        except Exception as e:
            msg = str(e)
            is_net_err = any(k in msg.lower() for k in [
                "timeout", "timed out", "connection", "dns", "ssl", "reset", "5xx",
                "service unavailable", "gateway", "network", "temporarily", "proxy"])
            is_quota = ("rate" in msg.lower()) or ("429" in msg) or ("quota" in msg.lower())
            log(f"⚠️ 调用失败（第{i + 1}/{max_retries}次）：{e}")

            # —— 全局错误计数（用于触发 STOP_ALL） ——
            # 记录当前错误时间戳并裁剪滑动窗口
            try:
                now = _now()
                _error_timestamps.append(now)
                cutoff = now - ERROR_WINDOW_S
                # pop 老的时间戳
                while _error_timestamps and _error_timestamps[0] < cutoff:
                    _error_timestamps.pop(0)
                if len(_error_timestamps) >= ERROR_THRESHOLD:
                    try:
                        STOP_ALL_FILE.write_text("error_threshold_reached\n", encoding="utf-8")
                        STOP_REASON_FILE.write_text(
                            f"[熔断] {len(_error_timestamps)} errors within {ERROR_WINDOW_S}s at {now}\n",
                            encoding="utf-8")
                    except Exception:
                        pass
                    print("🚨 触发 STOP_ALL：错误数超过阈值", flush=True)
                    # 触发全局熔断：抛出异常以便上层进程停止
                    raise RuntimeError("Global STOP_ALL triggered (too many errors)")
            except Exception:
                # 任何统计相关的小错误都不该阻止原有重试逻辑
                pass

            # 现有网络错误/熔断记录（保持你原来的逻辑）
            if is_net_err:
                record_net_error()
            # Half-open 阶段的一次成功会在外层通过 half_open_probe_ok() 关闭熔断
            if i == max_retries - 1:
                raise
            # 退避
            delay = min(base_delay * (2 ** i), DEFAULT_BACKOFF_MAX)
            time.sleep(delay)
            total_wait += delay
            if total_wait >= wait_budget:
                raise RuntimeError(f"WaitBudgetExceeded: 已等待 {total_wait}s，放弃重试")

@dataclass
class RunStatus:
    success: bool
    hard_fail: bool
    reason: Optional[str] = None
    detail: Optional[dict] = None

def save_run_status(book_id: str, step: str, status: RunStatus) -> Path:
    """
    将运行状态写入 results/.runtime/{book_id}_{step}_STATUS.json
    自动确保目录存在
    """
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = _STATUS_DIR / f"{book_id}_{step}_STATUS.json"
    path.write_text(json.dumps(asdict(status), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def delete_run_status(book_id: str, step: str) -> None:
    p = _STATUS_DIR / f"{book_id}_{step}_STATUS.json"
    if p.exists():
        p.unlink()
