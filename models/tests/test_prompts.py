# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import

from models.hybrid_search_retreiver import HybridSearchRetriever
from models.prompt_templates import NetecPromptTemplates


class TestPrompts:
    """Test HybridSearchRetriever class."""

    hsr = HybridSearchRetriever()
    templates = NetecPromptTemplates()

    def test_oracle_training_services(self):
        """Test a prompt with the Oracle training services template"""

        prompt = self.templates.oracle_training_services
        result = self.hsr.prompt_with_template(prompt=prompt, concept="Oracle database administrator")
        assert result
        assert "Oracle" in result
        assert "training" in result

    def test_training_services(self):
        """Test a prompt with the training services template"""

        prompt = self.templates.training_services
        result = self.hsr.prompt_with_template(prompt=prompt, concept="Microsoft certified Azure AI engineer associate")
        assert result
        assert "Microsoft" in result or "Azure" in result
        assert "training" in result
