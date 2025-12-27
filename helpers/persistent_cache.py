"""
Persistent Cache Implementation
Provides file-based caching with TTL and graceful degradation.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Callable
from pathlib import Path
import threading


class PersistentCache:
    """
    File-based cache with TTL support and thread-safe operations.
    
    Features:
    - Persistent storage to survive app restarts
    - TTL-based expiration
    - Thread-safe read/write
    - Graceful degradation on errors
    """
    
    def __init__(self, cache_dir: str = "/tmp/waves_cache", default_ttl: int = 3600):
        """
        Initialize persistent cache.
        
        Args:
            cache_dir: Directory for cache files
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't create cache dir, operations will degrade gracefully
            pass
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Hash the key to create a safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get the metadata file path for a cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            try:
                cache_path = self._get_cache_path(key)
                meta_path = self._get_metadata_path(key)
                
                # Check if cache file exists
                if not cache_path.exists() or not meta_path.exists():
                    return None
                
                # Read metadata
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check expiration
                expiry_time = datetime.fromisoformat(metadata['expiry'])
                if datetime.now() > expiry_time:
                    # Expired, clean up
                    cache_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return None
                
                # Read cached value
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                return value
                
            except Exception:
                # On any error, return None (cache miss)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if cached successfully, False otherwise
        """
        with self.lock:
            try:
                cache_path = self._get_cache_path(key)
                meta_path = self._get_metadata_path(key)
                
                # Calculate expiry time
                ttl = ttl or self.default_ttl
                expiry = datetime.now() + timedelta(seconds=ttl)
                
                # Write metadata
                metadata = {
                    'key': key,
                    'created': datetime.now().isoformat(),
                    'expiry': expiry.isoformat(),
                    'ttl': ttl
                }
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                # Write cached value
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                return True
                
            except Exception:
                # On any error, fail gracefully
                return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted successfully
        """
        with self.lock:
            try:
                cache_path = self._get_cache_path(key)
                meta_path = self._get_metadata_path(key)
                
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                
                return True
                
            except Exception:
                return False
    
    def clear(self) -> bool:
        """
        Clear all cached values.
        
        Returns:
            True if cleared successfully
        """
        with self.lock:
            try:
                for file in self.cache_dir.glob("*.cache"):
                    file.unlink(missing_ok=True)
                for file in self.cache_dir.glob("*.meta"):
                    file.unlink(missing_ok=True)
                return True
            except Exception:
                return False
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            try:
                cache_files = list(self.cache_dir.glob("*.cache"))
                meta_files = list(self.cache_dir.glob("*.meta"))
                
                total_size = sum(f.stat().st_size for f in cache_files)
                
                # Count expired entries
                expired_count = 0
                for meta_file in meta_files:
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        expiry_time = datetime.fromisoformat(metadata['expiry'])
                        if datetime.now() > expiry_time:
                            expired_count += 1
                    except Exception:
                        pass
                
                return {
                    'total_entries': len(cache_files),
                    'expired_entries': expired_count,
                    'total_size_bytes': total_size,
                    'cache_dir': str(self.cache_dir)
                }
            except Exception:
                return {
                    'total_entries': 0,
                    'expired_entries': 0,
                    'total_size_bytes': 0,
                    'cache_dir': str(self.cache_dir)
                }
    
    def cached_call(
        self,
        key: str,
        func: Callable,
        *args,
        ttl: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Get cached value or execute function and cache result.
        
        Args:
            key: Cache key
            func: Function to call if cache miss
            *args: Positional arguments for func
            ttl: Time-to-live in seconds
            **kwargs: Keyword arguments for func
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Cache miss, execute function
        result = func(*args, **kwargs)
        
        # Cache the result
        self.set(key, result, ttl=ttl)
        
        return result


# Global cache instance
_global_cache: Optional[PersistentCache] = None


def get_persistent_cache() -> PersistentCache:
    """Get or create the global persistent cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PersistentCache()
    return _global_cache
