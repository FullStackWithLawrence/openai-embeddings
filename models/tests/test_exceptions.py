# -*- coding: utf-8 -*-
"""Test models.exceptions."""

import pytest

from models.exceptions import (
    EXCEPTION_MAP,
    ConfigurationError,
    ModelConfigurationError,
    ModelIlligalInvocationError,
    ModelValueError,
    OpenAIResponseCodes,
)


class TestExceptions:
    """Test the project's exception classes."""

    @pytest.mark.parametrize(
        "exception_class",
        [ConfigurationError, ModelConfigurationError, ModelIlligalInvocationError, ModelValueError],
    )
    def test_message(self, exception_class):
        """Each exception keeps its message, and raises as itself."""
        with pytest.raises(exception_class) as excinfo:
            raise exception_class("test message")
        assert excinfo.value.message == "test message"
        assert str(excinfo.value) == "test message"

    def test_exception_map(self):
        """Every mapped exception has an http status code and a name."""
        for status, name in EXCEPTION_MAP.values():
            assert status in (
                OpenAIResponseCodes.HTTP_RESPONSE_BAD_REQUEST,
                OpenAIResponseCodes.HTTP_RESPONSE_INTERNAL_SERVER_ERROR,
            )
            assert name in ("BadRequest", "InternalServerError")
