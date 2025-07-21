from concurrent.futures import CancelledError, ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import multiprocessing
import os
from multiprocessing import get_context
import queue

from . import job_util

have_cloudpickle = False
cloudpickle = None
try:
    import cloudpickle

    have_cloudpickle = True
except ImportError:
    try:
        from joblib.externals import cloudpickle

        have_cloudpickle = True
    except ImportError:
        pass


class ThreadPoolExecutor(_ThreadPoolExecutor):
    """shim for same api ([mp_]context args)"""

    def __init__(
        self,
        max_workers=None,
        mp_context=None,
        initializer=None,
        initargs=None,
        context=None,
    ):
        super().__init__(
            max_workers=max_workers, initializer=initializer, initargs=initargs
        )


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
        self.initargs = initargs
        self.initializer = initializer
        self.initialized = False
        self.cancelled = False

    def _initialize(self):
        if self.initialized:
            return
        if self.initializer is not None:
            self.initializer(*self.initargs)
        self.initialized = True

    def map(self, function, *iterables, timeout=None, chunksize=1):
        self._initialize()
        for result in map(function, *iterables):
            yield result
            if self.cancelled:
                raise CancelledError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def submit(self, f, *args):
        self._initialize()
        return MockFuture(f, *args)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            self.cancelled = True


class MockQueue:
    """Another helper class for turning off concurrency when debugging."""

    def __init__(self):
        self.q = []
        self.put = self.q.append
        self.get = lambda: self.q.pop(0)


def cloudpickle_run(fn, args):
    assert cloudpickle is not None
    fn = cloudpickle.loads(fn)
    args, kwargs = cloudpickle.loads(args)
    # return cloudpickle.dumps(fn(*args, **kwargs))
    return fn(*args, **kwargs)


# def uncloudpickle(future):
#     res = future.result()
#     future._result = cloudpickle.loads(res)


class CloudpicklePoolExecutor(ProcessPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        assert cloudpickle is not None
        args = cloudpickle.dumps((args, kwargs))
        future = super().submit(cloudpickle_run, cloudpickle.dumps(fn), args)
        # future.add_done_callback(uncloudpickle_callback)
        return future


def rank_init(queue):
    rank_init.rank = queue.get()


def pool_from_cfg(computation_config=None, with_rank_queue=False, check_local=False):
    if computation_config is None:
        computation_config = job_util.get_global_computation_config()

    return get_pool(
        computation_config.actual_n_jobs(),
        cls=computation_config.executor,
        with_rank_queue=with_rank_queue,
        check_local=check_local,
        multi_gpu=computation_config.is_multi_gpu(),
    )


def get_pool(
    n_jobs,
    context="spawn",
    cls="threading_unless_multigpu",
    with_rank_queue=False,
    rank_queue_empty=False,
    n_tasks=None,
    max_tasks_per_child=None,
    check_local=False,
    multi_gpu=False,
):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    do_parallel = n_jobs >= 1
    n_jobs = max(1, n_jobs)

    if isinstance(cls, str):
        if cls == "threading_unless_multigpu":
            if (n_jobs > 1 and multi_gpu) or os.name == 'nt':
                cls = "ProcessPoolExecutor"
            else:
                cls = "ThreadPoolExecutor"
        if cls == "CloudpicklePoolExecutor":
            cls = CloudpicklePoolExecutor
        elif cls == "ThreadPoolExecutor":
            cls = ThreadPoolExecutor
        elif cls == "ProcessPoolExecutor":
            cls = ProcessPoolExecutor
        elif cls == "MockPoolExecutor":
            cls = MockPoolExecutor
        else:
            assert False

    if cls == CloudpicklePoolExecutor and not have_cloudpickle:
        cls = ProcessPoolExecutor

    Executor = cls if do_parallel else MockPoolExecutor
    is_local = cls in (MockPoolExecutor, ThreadPoolExecutor)
    context = get_context(context)

    if with_rank_queue:
        if do_parallel and is_local:
            rank_queue = queue.SimpleQueue()
        elif do_parallel:
            manager = context.Manager()
            rank_queue = manager.Queue()
        else:
            rank_queue = MockQueue()

        if not rank_queue_empty:
            n_repeats = 1
            if max_tasks_per_child is not None:
                n_repeats = n_tasks // max_tasks_per_child + 1
            for _ in range(n_repeats):
                for rank in range(n_jobs):
                    rank_queue.put(rank)

        if check_local:
            return n_jobs, Executor, context, rank_queue, is_local
        return n_jobs, Executor, context, rank_queue

    if check_local:
        return n_jobs, Executor, context, is_local
    return n_jobs, Executor, context
