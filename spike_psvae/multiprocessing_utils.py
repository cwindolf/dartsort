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
    ):
        initializer(*initargs)
        self.map = map

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
