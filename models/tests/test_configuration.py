# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test conf module.
"""
import os
from unittest.mock import PropertyMock, patch

import pytest  # pylint: disable=unused-import
from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic import ValidationError as PydanticValidationError

from models.conf import (
    Settings,
    SettingsDefaults,
    empty_str_to_bool_default,
    empty_str_to_int_default,
    load_version,
)
from models.exceptions import ModelConfigurationError, ModelValueError


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
        assert mock_settings.openai_chat_model_name == "TEST_gpt-4"
        assert mock_settings.openai_prompt_model_name == "TEST_gpt-4"
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
            openai_chat_model_name="TEST_gpt-4",
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
        assert mock_settings.openai_chat_model_name == "TEST_gpt-4"
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
            mock_settings.openai_chat_model_name = "TEST_gpt-4"
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


class TestConfigHelpers:
    """Test the module-level helpers and validators of models.conf, offline."""

    # Pydantic turns each validator into a classmethod, which pylint reads as an unbound method.
    # pylint: disable=no-value-for-parameter

    def test_load_version_missing_spec(self):
        """Load_version() raises ModelConfigurationError when the version file can't be loaded."""
        with patch("models.conf.importlib.util.spec_from_file_location", return_value=None):
            with pytest.raises(ModelConfigurationError):
                load_version()

    def test_settings_defaults_to_dict(self):
        """SettingsDefaults.to_dict() returns the defaults, without dunders or itself."""
        defaults = SettingsDefaults.to_dict()
        assert defaults["LANGCHAIN_MEMORY_KEY"] == SettingsDefaults.LANGCHAIN_MEMORY_KEY
        assert "to_dict" not in defaults
        assert not any(key.startswith("__") for key in defaults)

    @pytest.mark.parametrize(
        "value,default,expected",
        [(None, True, True), ("", False, False), ("yes", False, True), ("no", True, False)],
    )
    def test_empty_str_to_bool_default(self, value, default, expected):
        """Empty values return the default, others are parsed as booleans."""
        assert empty_str_to_bool_default(value, default) is expected

    @pytest.mark.parametrize("value,default,expected", [(None, 7, 7), ("", 7, 7), ("3", 7, 3), ("abc", 7, 7)])
    def test_empty_str_to_int_default(self, value, default, expected):
        """Empty or non-numeric values return the default, others are parsed as integers."""
        assert empty_str_to_int_default(value, default) == expected

    def test_api_key_sources_from_init_arguments(self):
        """Keys passed to the constructor are reported as init arguments."""
        with patch.dict(os.environ, {}, clear=True):
            mock_settings = Settings(openai_api_key="sk-test", pinecone_api_key="pc-test")
        assert mock_settings.openai_api_key_source == "init argument"
        assert mock_settings.pinecone_api_key_source == "init argument"

    def test_api_key_sources_from_environment(self):
        """Keys in the environment are reported as environment variables."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "PINECONE_API_KEY": "pc-test"}, clear=True):
            mock_settings = Settings()
        assert mock_settings.openai_api_key_source == "environment variable"
        assert mock_settings.pinecone_api_key_source == "environment variable"

    def test_placeholder_properties(self):
        """The tfvars and AWS properties are placeholders for future use."""
        mock_settings = Settings()
        assert not mock_settings.tfvars_variables
        assert mock_settings.is_using_aws_rekognition is False
        assert mock_settings.is_using_aws_dynamodb is False
        assert isinstance(mock_settings.environment_variables, list)

    def test_dump_is_cached(self):
        """Dump is computed once, then returned from the cache."""
        mock_settings = Settings()
        first = mock_settings.dump
        assert mock_settings.dump is first

    def test_dump_optional_sections(self):
        """Dump includes the defaults, dotenv and tfvars sections when they apply."""
        with (
            patch.object(Settings, "is_using_tfvars_file", new_callable=PropertyMock, return_value=True),
            patch.object(Settings, "is_using_dotenv_file", new_callable=PropertyMock, return_value=True),
        ):
            dump = Settings(dump_defaults=True).dump
        assert "settings_defaults" in dump
        assert dump["environment"]["tfvars"] == []
        assert isinstance(dump["environment"]["dotenv"], list)

    @pytest.mark.parametrize("validator", ["parse_debug_mode", "parse_dump_defaults", "check_openai_chat_cache"])
    def test_bool_validators(self, validator):
        """Boolean validators pass booleans through, default empty values, and parse strings."""
        func = getattr(Settings, validator)
        assert func(True) is True
        assert isinstance(func(None), bool)
        assert isinstance(func(""), bool)
        assert func("yes") is True
        assert func("no") is False

    @pytest.mark.parametrize(
        "validator,default",
        [
            ("check_langchain_memory_key", SettingsDefaults.LANGCHAIN_MEMORY_KEY),
            ("check_openai_endpoint_image_size", SettingsDefaults.OPENAI_ENDPOINT_IMAGE_SIZE),
            ("check_openai_chat_model_name", SettingsDefaults.OPENAI_CHAT_MODEL_NAME),
            ("check_openai_prompt_model_name", SettingsDefaults.OPENAI_PROMPT_MODEL_NAME),
            ("check_pinecone_environment", SettingsDefaults.PINECONE_ENVIRONMENT),
            ("check_pinecone_index_name", SettingsDefaults.PINECONE_INDEX_NAME),
            ("check_pinecone_vectorstore_text_key", SettingsDefaults.PINECONE_VECTORSTORE_TEXT_KEY),
            ("check_pinecone_metric", SettingsDefaults.PINECONE_METRIC),
        ],
    )
    def test_str_validators(self, validator, default):
        """String validators default empty values and pass others through."""
        func = getattr(Settings, validator)
        assert func(None) == default
        assert func("") == default
        assert func("TEST_value") == "TEST_value"

    @pytest.mark.parametrize(
        "validator,default,cast",
        [
            ("check_openai_endpoint_image_n", SettingsDefaults.OPENAI_ENDPOINT_IMAGE_N, int),
            ("check_openai_chat_temperature", SettingsDefaults.OPENAI_CHAT_TEMPERATURE, float),
            ("check_openai_chat_max_retries", SettingsDefaults.OPENAI_CHAT_MAX_RETRIES, int),
            ("check_pinecone_dimensions", SettingsDefaults.PINECONE_DIMENSIONS, int),
        ],
    )
    def test_numeric_validators(self, validator, default, cast):
        """Numeric validators default empty values and cast strings."""
        func = getattr(Settings, validator)
        assert func(None) == default
        assert func("") == default
        assert func("2") == cast("2")

    def test_check_openai_endpoint_image_n_int(self):
        """Integers pass through check_openai_endpoint_image_n unchanged."""
        assert Settings.check_openai_endpoint_image_n(9) == 9

    def test_check_openai_api_organization(self):
        """The organization is stripped, falls back to the default, or is None."""
        assert Settings.check_openai_api_organization("  org-test  ") == "org-test"
        with patch.object(SettingsDefaults, "OPENAI_API_ORGANIZATION", " org-default "):
            assert Settings.check_openai_api_organization("") == "org-default"
        with patch.object(SettingsDefaults, "OPENAI_API_ORGANIZATION", None):
            assert Settings.check_openai_api_organization(None) is None

    def test_check_openai_api_key(self):
        """The OpenAI key is required, and is always returned as a SecretStr."""
        secret = SecretStr("sk-test")
        assert Settings.check_openai_api_key(secret) is secret
        assert Settings.check_openai_api_key("sk-test").get_secret_value() == "sk-test"
        with pytest.raises(ModelValueError):
            Settings.check_openai_api_key(None)
        with patch.object(SettingsDefaults, "OPENAI_API_KEY", secret):
            assert Settings.check_openai_api_key(123) is secret
        with patch.object(SettingsDefaults, "OPENAI_API_KEY", None):
            with pytest.raises(ModelValueError):
                Settings.check_openai_api_key(123)

    def test_check_pinecone_api_key(self):
        """The Pinecone key falls back to the default, and is required."""
        secret = SecretStr("pc-test")
        assert Settings.check_pinecone_api_key(secret) is secret
        with patch.object(SettingsDefaults, "PINECONE_API_KEY", secret):
            assert Settings.check_pinecone_api_key(None) is secret
        with patch.object(SettingsDefaults, "PINECONE_API_KEY", None):
            with pytest.raises(ModelValueError):
                Settings.check_pinecone_api_key(None)
