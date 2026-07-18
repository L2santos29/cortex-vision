"""Unit tests for the profiling module."""

import asyncio
from unittest.mock import patch

import pytest

from src.profiling import timed, timed_context


class TestTimedDecorator:

    def test_timed_sync_decorator_returns_value(self):
        @timed(name="test_func")
        def my_func():
            return 42

        result = my_func()
        assert result == 42

    def test_timed_no_name_uses_qualname(self):
        @timed()
        def my_func():
            return "hello"

        assert my_func() == "hello"

    def test_timed_warn_threshold_triggers_warning(self):
        with patch("src.profiling.logger.warning") as mock_warn:
            @timed(warn_threshold=0.001)
            def slow_func():
                import time
                time.sleep(0.01)
                return "slow"

            result = slow_func()
            assert result == "slow"
            mock_warn.assert_called_once()
            # mock_warn.call_args[0] = (msg, *args), msg should contain "Profile [%s]"
            call_msg = mock_warn.call_args[0][0]
            assert "Profile [%s]" in call_msg

    def test_timed_sync_no_threshold_logs_debug(self):
        with patch("src.profiling.logger.log") as mock_log:
            @timed()
            def my_func():
                return 42

            my_func()
            mock_log.assert_called_once()

    def test_timed_async_decorator(self):
        @timed(name="async_func")
        async def async_func():
            return "async result"

        result = asyncio.run(async_func())
        assert result == "async result"

    def test_timed_async_decorator_logs(self):
        with patch("src.profiling.logger.log") as mock_log:
            @timed(name="async_func")
            async def async_func():
                return "async result"

            result = asyncio.run(async_func())
            assert result == "async result"
            mock_log.assert_called_once()


class TestTimedContext:

    def test_timed_context_returns_self(self):
        with timed_context("my_block") as ctx:
            pass
        assert ctx.name == "my_block"

    def test_timed_context_records_elapsed(self):
        with timed_context("measure") as ctx:
            import time
            time.sleep(0.005)
        assert ctx.elapsed > 0.004
        assert ctx.elapsed < 1.0

    def test_timed_context_warn_threshold_triggers_warning(self):
        with patch("src.profiling.logger.warning") as mock_warn:
            with timed_context("slow_block", warn_threshold=0.001):
                import time
                time.sleep(0.01)
            mock_warn.assert_called_once()
            call_msg = mock_warn.call_args[0][0]
            assert "Profile [%s]" in call_msg

    def test_timed_context_default_level_logs(self):
        with patch("src.profiling.logger.log") as mock_log:
            with timed_context("test"):
                pass
            mock_log.assert_called_once()

    def test_timed_context_no_threshold_logs_debug(self):
        with patch("src.profiling.logger.log") as mock_log:
            with timed_context("no_warn"):
                pass
            mock_log.assert_called_once()
