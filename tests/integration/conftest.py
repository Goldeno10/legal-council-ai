"""
Integration test configuration fixtures.
"""

import pytest
from src.api_client import MyClient


# Fixtures available for all integration tests
# Fixture Scope Examples:
# - function: runs once per test function (default)
# - class: runs once per test class
# - module: runs once per test module
# - session: runs once per test session

@pytest.fixture(scope="session")
def client():
    """Create a single client instance for the entire test session."""
    return MyClient(api_key="test_key")

@pytest.fixture
def mock_user_data():
    """Standard mock data used across multiple test files."""
    return {"id": 1, "name": "Test User"}
