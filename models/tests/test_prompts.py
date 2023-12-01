# -*- coding: utf-8 -*-
# flake8: noqa: F401
# pylint: disable=too-few-public-methods
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import

from ..prompt_templates import NetecPromptTemplates
from ..ssm import SalesSupportModel


class TestPrompts:
    """Test SalesSupportModel class."""

    ssm = SalesSupportModel()
    templates = NetecPromptTemplates()

    def test_oracle_training_services(self):
        """Test a prompt with the Oracle training services template"""

        prompt = self.templates.oracle_training_services
        result = self.ssm.prompt_with_template(prompt=prompt, concept="Oracle database administrator")
        assert result
        assert "Oracle" in result
        assert "training" in result

    def test_training_services(self):
        """Test a prompt with the training services template"""

        prompt = self.templates.training_services
        result = self.ssm.prompt_with_template(prompt=prompt, concept="Microsoft certified Azure AI engineer associate")
        assert result
        assert "Microsoft" in result
        assert "training" in result
