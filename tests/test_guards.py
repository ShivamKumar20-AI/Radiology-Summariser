import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guards import MAX_FINDINGS_CHARS, PerKeyLimiter, SlidingWindowLimiter, client_address


def test_window_blocks_after_limit_and_recovers():
    limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)
    assert [limiter.allow(now=t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert limiter.allow(now=61) is True


def test_each_address_has_its_own_allowance():
    limiter = PerKeyLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("a", now=0) and limiter.allow("a", now=1)
    assert not limiter.allow("a", now=2)
    assert limiter.allow("b", now=2)  # another visitor is not affected


def test_quiet_addresses_are_forgotten_so_memory_stays_bounded():
    limiter = PerKeyLimiter(max_requests=1, window_seconds=60, max_keys=3)
    for i, key in enumerate(["a", "b", "c"]):
        limiter.allow(key, now=i)
    limiter.allow("d", now=1000)  # triggers cleanup of the old, idle keys
    assert len(limiter._limiters) <= 2


def test_client_address_prefers_the_fly_header():
    assert client_address({"fly-client-ip": "203.0.113.7"}, "10.0.0.1") == "203.0.113.7"
    assert client_address({}, "10.0.0.1") == "10.0.0.1"


def test_size_limit_matches_the_web_page():
    assert MAX_FINDINGS_CHARS == 5000
    assert "/ 5000" in (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text()
