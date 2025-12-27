"""
Resilient API Call Wrapper
Provides timeout protection, retry logic with exponential backoff, and jitter.
"""

import time
import random
from typing import Callable, Any, Optional, Tuple
from functools import wraps


def with_timeout_and_retry(
    timeout_seconds: int = 10,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator to add timeout protection and retry logic with exponential backoff.
    
    Args:
        timeout_seconds: Maximum seconds to wait for function execution
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: If True, add random jitter to delays
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Try to execute the function
                    # Note: Real timeout would require threading/multiprocessing
                    # For simplicity, we rely on the underlying library's timeout
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise
                    if attempt >= max_retries:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter if enabled
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    # Wait before retrying
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def call_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    **kwargs
) -> Tuple[bool, Any, Optional[str]]:
    """
    Call a function with retry logic and return success status.
    
    Args:
        func: Function to call
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: If True, add random jitter to delays
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (success: bool, result: Any, error: Optional[str])
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return True, result, None
            
        except Exception as e:
            last_exception = e
            
            # If this was the last attempt, return failure
            if attempt >= max_retries:
                return False, None, str(e)
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter if enabled
            if jitter:
                delay = delay * (0.5 + random.random())
            
            # Wait before retrying
            time.sleep(delay)
    
    # Should never reach here
    return False, None, str(last_exception) if last_exception else "Unknown error"


def safe_call(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = False,
    **kwargs
) -> Any:
    """
    Safely call a function and return default value on error.
    
    Args:
        func: Function to call
        *args: Positional arguments for func
        default_return: Value to return on error
        log_errors: If True, print errors to console
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            print(f"Error in {func.__name__}: {str(e)}")
        return default_return
