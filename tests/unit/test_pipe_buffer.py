"""Tests for _PipeBuffer (producer/consumer thread-bridge with telemetry)."""

from __future__ import annotations

import threading
import time

import pytest

from oci_modelcar.runner import _PipeBuffer


def test_pipe_buffer_writable_and_get_chunk():
    """Smallest happy path: write some bytes, close, consume to EOF."""
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=16)
    assert pipe.writable() is True

    pipe.write(b"abcdefghijklmnop")  # exactly one coalesce chunk
    pipe.close()

    out = b""
    while True:
        chunk = pipe.get_chunk()
        if chunk is None:
            break
        out += chunk
    assert out == b"abcdefghijklmnop"


def test_pipe_buffer_coalesces_small_writes():
    """Small writes accumulate up to coalesce_size before hitting the queue."""
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10)
    pipe.write(b"abc")
    pipe.write(b"def")
    pipe.write(b"ghi")
    pipe.write(b"jkl")  # 12 bytes total; first 10 flush, 2 remain
    # one chunk should already be on the queue
    assert pipe._q.qsize() == 1
    pipe.close()  # flushes remainder + EOF
    chunks: list[bytes] = []
    while True:
        c = pipe.get_chunk()
        if c is None:
            break
        chunks.append(c)
    assert b"".join(chunks) == b"abcdefghijkl"


def test_pipe_buffer_write_returns_full_length():
    """Sink contract: write(n_bytes) returns n_bytes regardless of coalescing."""
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10)
    assert pipe.write(b"hello") == 5
    assert pipe.write(b"world!") == 6


def test_pipe_buffer_blocks_producer_when_full():
    """When the queue is full, producer .write() blocks until consumer drains.

    We exercise this by filling the queue, then having a consumer thread
    drain after a delay, and observing producer_wait_s reflects the wait.
    """
    pipe = _PipeBuffer(max_chunks=2, coalesce_size=10)

    def consumer() -> None:
        time.sleep(0.05)
        pipe.get_chunk()  # frees one slot

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    # First two writes fit; third blocks until consumer pops one
    pipe.write(b"a" * 10)
    pipe.write(b"b" * 10)
    pipe.write(b"c" * 10)  # blocks ~50 ms
    t.join(timeout=1.0)
    assert pipe.producer_wait_s >= 0.04


def test_pipe_buffer_consumer_blocks_when_empty():
    """When the queue is empty, consumer .get_chunk() blocks until producer
    writes; consumer_wait_s reflects the wait."""
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10)

    def producer() -> None:
        time.sleep(0.05)
        pipe.write(b"x" * 10)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    chunk = pipe.get_chunk()
    t.join(timeout=1.0)
    assert chunk == b"x" * 10
    assert pipe.consumer_wait_s >= 0.04


def test_pipe_buffer_bytes_through_counts_consumed_only():
    """bytes_through tracks bytes pulled by the consumer, not bytes written.

    Coalesced + still in producer-side buffer don't count yet.
    """
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10)
    pipe.write(b"a" * 10)
    pipe.write(b"b" * 5)  # 5 bytes still in pre-coalesce buffer
    pipe.close()
    while pipe.get_chunk() is not None:
        pass
    assert pipe.bytes_through == 15


def test_pipe_buffer_get_chunk_returns_none_after_eof():
    """Once EOF is consumed, further get_chunk calls return None idempotently
    only if the producer keeps closing — but a clean post-EOF state should
    not crash. Single EOF is the supported contract."""
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10)
    pipe.write(b"hello")
    pipe.close()
    assert pipe.get_chunk() == b"hello"
    assert pipe.get_chunk() is None


def test_pipe_buffer_write_after_close_is_a_bug_we_dont_protect_against():
    """Documenting the contract: producer must not call write() after close().
    The class doesn't enforce this — it's the caller's discipline."""
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10)
    pipe.close()
    # No assertion about behavior; just documenting the contract.
    pytest.skip("contract documentation only")


def test_pipe_buffer_propagates_producer_exception():
    """Producer's report_exception is surfaced from get_chunk on the consumer."""

    class BoomError(RuntimeError):
        pass

    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10)
    pipe.write(b"a" * 10)
    pipe.report_exception(BoomError("upstream HF cut"))

    # First chunk delivers normally
    assert pipe.get_chunk() == b"a" * 10
    # Then the producer's exception surfaces
    with pytest.raises(BoomError, match="upstream HF cut"):
        pipe.get_chunk()


def test_pipe_buffer_write_aborts_on_stop_event():
    """If the external stop_event is set, write() raises InterruptedError."""
    stop = threading.Event()
    pipe = _PipeBuffer(max_chunks=4, coalesce_size=10, stop_event=stop)
    pipe.write(b"a" * 5)  # fits, doesn't trigger put
    stop.set()
    with pytest.raises(InterruptedError):
        pipe.write(b"b" * 10)


def test_pipe_buffer_drain_and_abort_unblocks_producer():
    """After consumer calls drain_and_abort, a producer blocked on put() is
    unstuck on the next iteration: drain frees slots, the in-flight put
    returns, and the next write() raises InterruptedError."""
    pipe = _PipeBuffer(max_chunks=2, coalesce_size=10)

    error: list[BaseException] = []

    def producer() -> None:
        try:
            for i in range(10):
                pipe.write(bytes([ord("a") + i]) * 10)
        except BaseException as e:
            error.append(e)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    # Let producer fill the queue and block on put
    time.sleep(0.05)
    assert pipe._q.qsize() == 2

    pipe.drain_and_abort()
    t.join(timeout=2.0)
    assert not t.is_alive(), "producer should have exited after drain_and_abort"
    assert error and isinstance(error[0], InterruptedError), error
