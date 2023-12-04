# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test command line example prompts.
"""
from unittest.mock import MagicMock, patch

import pytest  # pylint: disable=unused-import
from langchain.schema import HumanMessage, SystemMessage

from models.examples.certification_programs import hsr as uofpenn_certification_program
from models.examples.online_courses import hsr as uofpenn_online_hsr
from models.examples.prompt import hsr as prompt_hrs
from models.examples.rag import hsr as rag_hsr
from models.prompt_templates import NetecPromptTemplates


HUMAN_MESSAGE = 'return the word "SUCCESS" in upper case.'


class TestExamples:
    """Test command line examples."""

    @patch("argparse.ArgumentParser.parse_args")
    def test_prompt(self, mock_parse_args):
        """Test prompt example."""
        mock_args = MagicMock()
        mock_args.system_prompt = "you are a helpful assistant"
        mock_args.human_prompt = HUMAN_MESSAGE
        mock_parse_args.return_value = mock_args

        system_message = SystemMessage(content="you are a helpful assistant")
        human_message = HumanMessage(content=HUMAN_MESSAGE)
        result = prompt_hrs.cached_chat_request(system_message=system_message, human_message=human_message)
        assert result.content == "SUCCESS"

    @patch("argparse.ArgumentParser.parse_args")
    def test_rag(self, mock_parse_args):
        """Test RAG example."""
        mock_args = MagicMock()
        mock_args.human_message = HUMAN_MESSAGE
        mock_parse_args.return_value = mock_args

        human_message = HumanMessage(content=mock_args.human_message)
        result = rag_hsr.rag(human_message=human_message)
        assert result == "SUCCESS"

    @patch("argparse.ArgumentParser.parse_args")
    def test_training_services(self, mock_parse_args):
        """Test training services templates."""
        mock_args = MagicMock()
        mock_args.human_message = HUMAN_MESSAGE
        mock_parse_args.return_value = mock_args

        templates = NetecPromptTemplates()
        prompt = templates.training_services

        result = uofpenn_certification_program.prompt_with_template(prompt=prompt, concept=mock_args.human_message)
        assert "SUCCESS" in result

    @patch("argparse.ArgumentParser.parse_args")
    def test_oracle_training_services(self, mock_parse_args):
        """Test oracle training services."""
        mock_args = MagicMock()
        mock_args.human_message = HUMAN_MESSAGE
        mock_parse_args.return_value = mock_args

        templates = NetecPromptTemplates()
        prompt = templates.oracle_training_services

        result = uofpenn_online_hsr.prompt_with_template(prompt=prompt, concept=mock_args.human_message)
        assert "SUCCESS" in result
