import copy
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Hashable


class _Entry:

    # Max lifetime of an entry in the cache in seconds.
    _default_ttl = 3600

    def __init__(self, value: Any, ttl: int = _default_ttl):
        self._value = value
        self._created_at = time.monotonic()
        self._ttl = ttl

    @property
    def value(self):
        return copy.deepcopy(self._value)

    def has_expired(self):
        delta_t = time.monotonic() - self._created_at
        return delta_t > self._ttl


class Cache:
    """Thread safe key-value cache."""

    # Max items in the client cache.
    _default_size = 1000

    def __init__(self, max_size: int = _default_size):
        """Initialize a Cache instance.

        Args:
            max_size: The maximum size of the cache. Defaults to the cache_size
                value in configs. Positive integer.

        Raises:
            TypeError if max_size is not an int.
            ValueError if max_size is not positive.

        """
        if not isinstance(max_size, int):
            raise TypeError(('max_size must be an integer, got type: '
                             '{}').format(type(max_size)))
        if max_size < 1:
            raise ValueError('max_size must be positive, got value: {}'.format(
                max_size))

        self._cache = OrderedDict()
        self._lock = Lock()
        self._max_size = max_size

    def get(self, key: Hashable) -> Any:
        """Get a value from the cache if it exists.

        Args:
            key: The cache key.

        Returns:
            The value associated with the key if it is in the cache.

        Raises:
            KeyError if the key is not in the cache.

        """
        with self._lock:
            entry = self._cache[key]
            if entry.has_expired():
                del self._cache[key]
                raise KeyError(key)
            else:
                return entry.value

    def set(self, key: Hashable, value: Any):
        """Add a value to the cache or overwrite an existing one.

        Args:
            key: The cache key.
            value: The value to add to the cache. The value must support
                __deepcopy__.

        """
        with self._lock:
            if len(self._cache) == self._max_size:
                # Remove the oldest item.
                self._cache.popitem(last=False)

            # Make sure key is inserted to the last position.
            if key in self._cache:
                del self._cache[key]

            self._cache[key] = _Entry(value)
