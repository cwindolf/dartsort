from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor


class MockFuture:
    """See below."""

    def __init__(self, f, *args):
        self.f = f
        self.args = args

    def result(self):
        return self.f(*self.args)


class MockPoolExecutor:
    """A helper class for turning off concurrency when debugging."""

    def __init__(
        self,
        max_workers=None,
        mp_context=None,
        initializer=None,
        initargs=None,
        context=None,
    ):
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


def get_pool(n_jobs, context="spawn", cls=ProcessPoolExecutor):
    Executor = cls if (n_jobs and n_jobs > 1) else MockPoolExecutor
    context = get_context(context)
    return Executor, context
