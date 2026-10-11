import pytest

from dartsort.util.multiprocessing_util import (
    MockPoolExecutor,
    ThreadPoolExecutor,
    buffered_map,
)


@pytest.mark.parametrize("Executor", [ThreadPoolExecutor, MockPoolExecutor])
def test_buffered_map(Executor):
    n_jobs = 50
    buffersize = 4
    n_submitted = 0

    def jobs():
        nonlocal n_submitted
        for j in range(n_jobs):
            n_submitted += 1
            yield j

    def square(j):
        return j * j

    results = []
    with Executor(max_workers=2) as pool:
        for r in buffered_map(pool, square, jobs(), buffersize=buffersize):
            results.append(r)
            assert n_submitted <= len(results) + buffersize
    assert results == [j * j for j in range(n_jobs)]
