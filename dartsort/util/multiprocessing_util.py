import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import cloudpickle


class MockFuture:
    """A non-concurrent class for mocking the concurrent.futures API."""

    def __init__(self, f, *args):
        self.f = f
        self.args = args

    def result(self):
        return self.f(*self.args)


class MockPoolExecutor:
    """A non-concurrent class for mocking the concurrent.futures API."""

    def __init__(
        self,
        max_workers=None,
        mp_context=None,
        initializer=None,
        initargs=None,
        context=None,
    ):
        if initializer is not None:
            initializer(*initargs)
        self.map = map
        self.imap = map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def submit(self, f, *args):
        return MockFuture(f, *args)


class MockQueue:
    """Another helper class for turning off concurrency when debugging."""

    def __init__(self):
        self.q = []
        self.put = self.q.append
        self.get = lambda: self.q.pop(0)


def apply_cloudpickle(fn, /, *args, **kwargs):
    fn = cloudpickle.loads(fn)
    return fn(*args, **kwargs)


class CloudpicklePoolExecutor(ProcessPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(
            apply_cloudpickle, cloudpickle.dumps(fn), *args, **kwargs
        )


def get_pool(
    n_jobs, context="spawn", cls=ProcessPoolExecutor, with_rank_queue=False
):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    do_parallel = n_jobs >= 1
    n_jobs = max(1, n_jobs)
    Executor = cls if do_parallel else MockPoolExecutor
    context = get_context(context)
    if with_rank_queue:
        if do_parallel:
            manager = context.Manager()
            rank_queue = manager.Queue()
        else:
            rank_queue = MockQueue()
        for rank in range(n_jobs):
            rank_queue.put(rank)
        return n_jobs, Executor, context, rank_queue
    return n_jobs, Executor, context
