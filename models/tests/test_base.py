# -*- coding: utf-8 -*-
# flake8: noqa: F401
# pylint: disable=too-few-public-methods
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import

from ..ssm import SalesSupportModel


class TestSalesSupportModel:
    """Test SalesSupportModel class."""

    def test_01_basic(self):
        """Test a basic request"""

        SalesSupportModel()
