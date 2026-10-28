# -*- coding: utf-8 -*-
# flake8: noqa: F401
# pylint: disable=too-few-public-methods
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import
from langchain_core.prompts import PromptTemplate

from models.prompt_templates import NetecPromptTemplates, UofPennPromptTemplates


class TestPromptTemplates:
    """Test HybridSearchRetriever class."""

    def test_01_prompt_with_template(self):
        """Ensure that all properties of the template class are PromptTemplate instances."""
        templates = NetecPromptTemplates()
        for prop_name in templates.get_properties():
            prop = getattr(templates, prop_name)
            assert isinstance(prop, PromptTemplate)

    def test_02_uofpenn_templates(self):
        """Every UofPenn template is a PromptTemplate that takes a concept."""
        templates = UofPennPromptTemplates()
        properties = templates.get_properties()
        assert properties
        for prop_name in properties:
            prop = getattr(templates, prop_name)
            assert isinstance(prop, PromptTemplate)
            assert "accounting" in prop.format(concept="accounting")
