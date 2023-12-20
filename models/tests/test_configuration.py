# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test conf module.
"""
import os
from unittest.mock import patch

import pytest  # pylint: disable=unused-import
from dotenv import load_dotenv
from pydantic import ValidationError as PydanticValidationError

from models.conf import Settings, SettingsDefaults


HERE = os.path.dirname(os.path.abspath(__file__))


class TestConfig:
    """Test config.settings."""

    def env_path(self, filename):
        """Return the path to the .env file."""
        return os.path.join(HERE, "mock_data", filename)

    def test_conf_defaults(self):
        """Test that settings == SettingsDefaults when no .env is in use."""
        os.environ.clear()
        mock_settings = Settings()
        assert mock_settings.langchain_memory_key == SettingsDefaults.LANGCHAIN_MEMORY_KEY
        assert mock_settings.debug_mode == SettingsDefaults.DEBUG_MODE

        assert mock_settings.openai_api_key == SettingsDefaults.OPENAI_API_KEY
        assert mock_settings.openai_api_organization == SettingsDefaults.OPENAI_API_ORGANIZATION
        assert mock_settings.openai_chat_cache == SettingsDefaults.OPENAI_CHAT_CACHE
        assert mock_settings.openai_chat_max_retries == SettingsDefaults.OPENAI_CHAT_MAX_RETRIES
        assert mock_settings.openai_chat_model_name == SettingsDefaults.OPENAI_CHAT_MODEL_NAME
        assert mock_settings.openai_endpoint_image_n == SettingsDefaults.OPENAI_ENDPOINT_IMAGE_N
        assert mock_settings.openai_endpoint_image_size == SettingsDefaults.OPENAI_ENDPOINT_IMAGE_SIZE
        assert mock_settings.openai_prompt_model_name == SettingsDefaults.OPENAI_PROMPT_MODEL_NAME

        assert mock_settings.pinecone_api_key == SettingsDefaults.PINECONE_API_KEY
        assert mock_settings.pinecone_dimensions == SettingsDefaults.PINECONE_DIMENSIONS
        assert mock_settings.pinecone_environment == SettingsDefaults.PINECONE_ENVIRONMENT
        assert mock_settings.pinecone_index_name == SettingsDefaults.PINECONE_INDEX_NAME
        assert mock_settings.pinecone_metric == SettingsDefaults.PINECONE_METRIC
        assert mock_settings.pinecone_vectorstore_text_key == SettingsDefaults.PINECONE_VECTORSTORE_TEXT_KEY

    # pylint: disable=no-member
    def test_conf_defaults_secrets(self):
        """Test that settings secrets match the defaults."""
        os.environ.clear()
        mock_settings = Settings()
        assert mock_settings.openai_api_key.get_secret_value() == SettingsDefaults.OPENAI_API_KEY.get_secret_value()
        assert mock_settings.pinecone_api_key.get_secret_value() == SettingsDefaults.PINECONE_API_KEY.get_secret_value()

    def test_env_legal_nulls(self):
        """Test that settings handles missing .env values."""
        os.environ.clear()
        env_path = self.env_path(".env.test_legal_nulls")
        print("env_path", env_path)
        loaded = load_dotenv(env_path)
        assert loaded

        mock_settings = Settings()
        assert mock_settings.langchain_memory_key == SettingsDefaults.LANGCHAIN_MEMORY_KEY
        assert mock_settings.openai_endpoint_image_size == SettingsDefaults.OPENAI_ENDPOINT_IMAGE_SIZE

    def test_env_illegal_nulls(self):
        """Test that settings handles missing .env values."""
        os.environ.clear()
        env_path = self.env_path(".env.test_illegal_nulls")
        print("env_path", env_path)
        loaded = load_dotenv(env_path)
        assert loaded

        with pytest.raises(PydanticValidationError):
            Settings()

    def test_env_overrides(self):
        """Test that settings takes custom .env values."""
        os.environ.clear()
        env_path = self.env_path(".env.test_01")
        loaded = load_dotenv(env_path)
        assert loaded

        mock_settings = Settings()

        assert mock_settings.debug_mode is True
        assert mock_settings.dump_defaults is True
        assert mock_settings.langchain_memory_key == "TEST_chat_history"
        assert mock_settings.pinecone_environment == "TEST_gcp-starter"
        assert mock_settings.pinecone_index_name == "TEST_rag"
        assert mock_settings.pinecone_vectorstore_text_key == "TEST_lc_id"
        assert mock_settings.pinecone_metric == "TEST_dotproduct"
        assert mock_settings.pinecone_dimensions == 1
        assert mock_settings.openai_endpoint_image_n == 1
        assert mock_settings.openai_endpoint_image_size == "TEST_1024x768"
        assert mock_settings.openai_chat_cache is False
        assert mock_settings.openai_chat_model_name == "TEST_gpt-3.5-turbo"
        assert mock_settings.openai_prompt_model_name == "TEST_text-davinci-003"
        assert mock_settings.openai_chat_temperature == 1.0
        assert mock_settings.openai_chat_max_retries == 5

    @patch.dict(os.environ, {"OPENAI_CHAT_MAX_RETRIES": "-1"})
    def test_invalid_chat_max_retries(self):
        """Test that Pydantic raises a validation error for environment variable w negative integer values."""

        with pytest.raises(PydanticValidationError):
            Settings()

    @patch.dict(os.environ, {"OPENAI_CHAT_TEMPERATURE": "-1"})
    def test_invalid_chat_temperature(self):
        """Test that Pydantic raises a validation error for environment variable w negative integer values."""

        with pytest.raises(PydanticValidationError):
            Settings()

    @patch.dict(os.environ, {"PINECONE_DIMENSIONS": "-1"})
    def test_invalid_pinecone_dimensions(self):
        """Test that Pydantic raises a validation error for environment variable w negative integer values."""

        with pytest.raises(PydanticValidationError):
            Settings()

    def test_configure_with_class_constructor(self):
        """test that we can set values with the class constructor"""
        os.environ.clear()

        mock_settings = Settings(
            debug_mode=True,
            dump_defaults=True,
            langchain_memory_key="TEST_chat_history",
            pinecone_environment="TEST_gcp-starter",
            pinecone_index_name="TEST_rag",
            pinecone_vectorstore_text_key="TEST_lc_id",
            pinecone_metric="TEST_dotproduct",
            pinecone_dimensions=1,
            openai_endpoint_image_n=1,
            openai_endpoint_image_size="TEST_1024x768",
            openai_chat_cache=False,
            openai_chat_model_name="TEST_gpt-3.5-turbo",
            openai_prompt_model_name="TEST_text-davinci-003",
            openai_chat_temperature=1.0,
            openai_chat_max_retries=5,
        )

        assert mock_settings.debug_mode is True
        assert mock_settings.dump_defaults is True
        assert mock_settings.langchain_memory_key == "TEST_chat_history"
        assert mock_settings.pinecone_environment == "TEST_gcp-starter"
        assert mock_settings.pinecone_index_name == "TEST_rag"
        assert mock_settings.pinecone_vectorstore_text_key == "TEST_lc_id"
        assert mock_settings.pinecone_metric == "TEST_dotproduct"
        assert mock_settings.pinecone_dimensions == 1
        assert mock_settings.openai_endpoint_image_n == 1
        assert mock_settings.openai_endpoint_image_size == "TEST_1024x768"
        assert mock_settings.openai_chat_cache is False
        assert mock_settings.openai_chat_model_name == "TEST_gpt-3.5-turbo"
        assert mock_settings.openai_prompt_model_name == "TEST_text-davinci-003"
        assert mock_settings.openai_chat_temperature == 1.0
        assert mock_settings.openai_chat_max_retries == 5

    def test_readonly_settings(self):
        """test that we can't set readonly values with the class constructor"""

        mock_settings = Settings()
        with pytest.raises(PydanticValidationError):
            mock_settings.langchain_memory_key = "TEST_chat_history"
        with pytest.raises(PydanticValidationError):
            mock_settings.pinecone_environment = "TEST_gcp-starter"
        with pytest.raises(PydanticValidationError):
            mock_settings.pinecone_index_name = "TEST_rag"
        with pytest.raises(PydanticValidationError):
            mock_settings.pinecone_vectorstore_text_key = "TEST_lc_id"
        with pytest.raises(PydanticValidationError):
            mock_settings.pinecone_metric = "TEST_dotproduct"
        with pytest.raises(PydanticValidationError):
            mock_settings.pinecone_dimensions = 1
        with pytest.raises(PydanticValidationError):
            mock_settings.openai_endpoint_image_n = 1
        with pytest.raises(PydanticValidationError):
            mock_settings.openai_endpoint_image_size = "TEST_1024x768"
        with pytest.raises(PydanticValidationError):
            mock_settings.openai_chat_cache = False
        with pytest.raises(PydanticValidationError):
            mock_settings.openai_chat_model_name = "TEST_gpt-3.5-turbo"
        with pytest.raises(PydanticValidationError):
            mock_settings.openai_prompt_model_name = "TEST_text-davinci-003"
        with pytest.raises(PydanticValidationError):
            mock_settings.openai_chat_temperature = 1.0
        with pytest.raises(PydanticValidationError):
            mock_settings.openai_chat_max_retries = 5

    def test_dump(self):
        """Test that dump is a dict."""

        mock_settings = Settings()
        assert isinstance(mock_settings.dump, dict)

    def test_dump_keys(self):
        """Test that dump contains the expected keys."""

        dump = Settings().dump
        assert "secrets" in dump.keys()
        assert "environment" in dump.keys()
        assert "langchain" in dump.keys()
        assert "openai_api" in dump.keys()
        assert "pinecone_api" in dump.keys()
