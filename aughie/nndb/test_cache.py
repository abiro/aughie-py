from functools import partial
from threading import Thread
from unittest import TestCase, mock

from .cache import Cache, _Entry


class TestCache(TestCase):
    def test_init(self):
        with self.assertRaises(TypeError):
            Cache(max_size=1.5)

        with self.assertRaises(ValueError):
            Cache(max_size=0)

    def test_ops(self):
        cache = Cache()
        cache.set('key', 'val')
        self.assertEqual(cache.get('key'), 'val')

    @mock.patch('aughie.nndb.cache.time')
    def test_ttl(self, mock_time):
        cache = Cache()
        ttl = _Entry._default_ttl
        self.assertGreater(ttl, 60)
        mock_time.monotonic.return_value = 0
        cache.set('key', 'val')
        mock_time.monotonic.return_value = ttl + 1
        mock_time.monotonic.assert_called_once_with()
        with self.assertRaises(KeyError):
            cache.get('key')

    def test_max_size(self):
        self.assertGreaterEqual(Cache._default_size, 100)

        max_size = 100
        cache = Cache(max_size=max_size)

        def run_test(start_i, n):
            for i in range(start_i, start_i + n):
                cache.set(i, i)
                self.assertLessEqual(len(cache._cache), max_size)

        n_tests = max_size * 100
        t1 = Thread(target=partial(run_test, 0, n_tests))
        t2 = Thread(target=partial(run_test, n_tests, n_tests))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def test_evict_order(self):
        cache = Cache()
        max_size = Cache._default_size

        for i in range(max_size - 1):
            cache.set(i, i)

        # Get key 1 evicted instead of 0 by bringing it to the front.
        cache.set(0, 0)
        cache.set(max_size - 1, max_size - 1)
        cache.set(max_size, max_size)

        with self.assertRaises(KeyError):
            cache.get(1)

        for i in range(max_size + 1):
            if i != 1:
                self.assertEqual(cache.get(i), i)

    def test_repeated_set(self):
        max_size = 100
        cache = Cache(max_size)
        # Fill up the cache and make key 0 the last added item.
        for i in range(max_size):
            cache.set(i, i)
        self.assertEqual(cache.get(0), 0)
        cache.set(0, 0)
        # Make key 0 the oldest item.
        for i in range(1, max_size):
            cache.set(i + max_size, i)
        self.assertEqual(cache.get(0), 0)
        # Make key 0 the last item.
        cache.set(0, 'new-val')
        self.assertEqual(cache.get(0), 'new-val')
        cache.set('foo', 'bar')
        self.assertEqual(cache.get(0), 'new-val')
        # Evict key 0 from the cache.
        for i in range(1, max_size):
            cache.set(i + 2 * max_size, i)
        with self.assertRaises(KeyError):
            cache.get(0)

    def test_deep_copy(self):
        cache = Cache()
        cache.set('key', [{'foo': 'bar'}])
        v = cache.get('key')
        v[0]['foo'] = 'baz'
        self.assertEqual(cache.get('key'), [{'foo': 'bar'}])
