import requests
import pytest

@pytest.fixture(scope="module")
def db_connection():
    # Setup: runs BEFORE the first test in the module
    conn = "Database Connected" 
    print("\n[Setup] Connecting to DB")
    yield conn 
    # Teardown: runs AFTER the last test in the module
    print("\n[Teardown] Closing DB connection")

def test_query(db_connection):
    assert db_connection == "Database Connected"

@pytest.mark.parametrize("input_val, expected", [
    (1, 2),
    (5, 6),
    (10, 11),
])
def test_increment(input_val, expected):
    assert input_val + 1 == expected


# Requires: pip install pytest-mock
def test_api_call(mocker):
    # Mock 'requests.get' so it doesn't actually hit the internet
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.status_code = 200

    response = requests.get("https://google.com")
    assert response.status_code == 200
