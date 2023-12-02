# -*- coding: utf-8 -*-
# flake8: noqa: F401
# pylint: disable=too-few-public-methods
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import

from models.hybrid_search_retreiver import HybridSearchRetriever


class TestOpenAI:
    """Test HybridSearchRetriever class."""

    def test_03_test_openai_connectivity(self):
        """Ensure that we have connectivity to OpenAI."""

        hsr = HybridSearchRetriever()
        retval = hsr.cached_chat_request(
            "your are a helpful assistant", "please return the value 'CORRECT' in all upper case."
        )
        assert retval.content == "CORRECT"
