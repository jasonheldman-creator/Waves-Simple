"""
Circuit Breaker Pattern Implementation
Provides resilience for external API calls (e.g., yfinance) with automatic recovery.
"""

import time
from enum import Enum
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timedelta
import threading


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    States:
    - CLOSED: Normal operation, calls proceed
    - OPEN: Too many failures, calls rejected immediately
    - HALF_OPEN: Testing if service recovered, limited calls allowed
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery (half-open)
        success_threshold: Number of successes in half-open to close circuit
        timeout: Maximum seconds to wait for a call to complete
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        timeout: int = 10
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
        
    def call(self, func: Callable, *args, **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Execute a function call through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (success: bool, result: Any, error: Optional[str])
        """
        with self.lock:
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        # Transition to half-open
                        self.state = CircuitState.HALF_OPEN
                        self.success_count = 0
                    else:
                        return False, None, f"Circuit breaker open, retry in {int(self.recovery_timeout - elapsed)}s"
                else:
                    return False, None, "Circuit breaker open"
            
            # State is CLOSED or HALF_OPEN, attempt the call
            current_state = self.state
        
        try:
            # Execute the function with timeout protection
            result = func(*args, **kwargs)
            
            # Record success
            with self.lock:
                if current_state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        # Recovery successful, close circuit
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        self.success_count = 0
                elif current_state == CircuitState.CLOSED:
                    # Reset failure count on success
                    self.failure_count = 0
            
            return True, result, None
            
        except Exception as e:
            error_msg = str(e)
            
            # Record failure
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if current_state == CircuitState.HALF_OPEN:
                    # Failed during recovery, reopen circuit
                    self.state = CircuitState.OPEN
                    self.success_count = 0
                elif self.failure_count >= self.failure_threshold:
                    # Too many failures, open circuit
                    self.state = CircuitState.OPEN
            
            return False, None, error_msg
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "is_available": self.state != CircuitState.OPEN
            }
    
    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None


# Global circuit breakers for different services
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create a named circuit breaker.
    
    Args:
        name: Unique name for the circuit breaker
        **kwargs: Configuration options for CircuitBreaker
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(**kwargs)
    return _circuit_breakers[name]


def get_all_circuit_states() -> Dict[str, Dict[str, Any]]:
    """Get the state of all circuit breakers."""
    return {name: cb.get_state() for name, cb in _circuit_breakers.items()}
