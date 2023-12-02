# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test command line example prompts.
"""
from unittest.mock import MagicMock, patch

import pytest  # pylint: disable=unused-import

from models.examples.prompt import hsr as prompt_hrs
from models.examples.rag import hsr as rag_hsr
from models.examples.training_services import hsr as training_services_hsr
from models.examples.training_services_oracle import hsr as training_services_oracle_hsr
from models.prompt_templates import NetecPromptTemplates


HUMAN_PROMPT = 'return the word "SUCCESS" in upper case.'


class TestExamples:
    """Test command line examples."""

    @patch("argparse.ArgumentParser.parse_args")
    def test_prompt(self, mock_parse_args):
        """Test prompt example."""
        mock_args = MagicMock()
        mock_args.system_prompt = "you are a helpful assistant"
        mock_args.human_prompt = HUMAN_PROMPT
        mock_parse_args.return_value = mock_args

        result = prompt_hrs.cached_chat_request(mock_args.system_prompt, mock_args.human_prompt)
        assert result == "SUCCESS"

    @patch("argparse.ArgumentParser.parse_args")
    def test_rag(self, mock_parse_args):
        """Test RAG example."""
        mock_args = MagicMock()
        mock_args.human_prompt = HUMAN_PROMPT
        mock_parse_args.return_value = mock_args

        result = rag_hsr.rag(mock_args.human_prompt)
        assert result == "SUCCESS"

    @patch("argparse.ArgumentParser.parse_args")
    def test_training_services(self, mock_parse_args):
        """Test training services templates."""
        mock_args = MagicMock()
        mock_args.human_prompt = HUMAN_PROMPT
        mock_parse_args.return_value = mock_args

        templates = NetecPromptTemplates()
        prompt = templates.training_services

        result = training_services_hsr.prompt_with_template(prompt=prompt, concept=mock_args.human_prompt)
        assert "SUCCESS" in result

    @patch("argparse.ArgumentParser.parse_args")
    def test_oracle_training_services(self, mock_parse_args):
        """Test oracle training services."""
        mock_args = MagicMock()
        mock_args.human_prompt = HUMAN_PROMPT
        mock_parse_args.return_value = mock_args

        templates = NetecPromptTemplates()
        prompt = templates.oracle_training_services

        result = training_services_oracle_hsr.prompt_with_template(prompt=prompt, concept=mock_args.human_prompt)
        assert "SUCCESS" in result
