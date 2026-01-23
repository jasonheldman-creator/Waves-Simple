from enum import Enum
from typing import Callable, Any, Optional, Dict
from datetime import datetime
import threading


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        timeout: int = 10,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                return False, None, "Circuit breaker open"

        try:
            result = func(*args, **kwargs)

            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        self.success_count = 0
                else:
                    self.failure_count = 0

            return True, result, None

        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
            return False, None, str(e)

    def reset(self) -> None:
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat()
                if self.last_failure_time
                else None,
                "is_available": self.state != CircuitState.OPEN,
            }


# ============================================================
# GLOBAL CIRCUIT BREAKER REGISTRY (CANONICAL)
# ============================================================

_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(**kwargs)
    return _circuit_breakers[name]


def get_all_circuit_states() -> Dict[str, Dict[str, Any]]:
    return {name: cb.get_state() for name, cb in _circuit_breakers.items()}


def reset_all_circuit_breakers() -> None:
    """
    Reset ALL registered circuit breakers.

    This is an OPERATIONAL utility.
    It must NEVER treat the registry as callable.
    """
    for cb in _circuit_breakers.values():
        cb.reset()