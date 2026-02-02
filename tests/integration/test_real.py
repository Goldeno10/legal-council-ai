# tests/test_api.py
def test_get_user_email_success(mocker, client, mock_user_data):
    # 1. Patch the actual network call
    # Note: Patch where the object is IMPORTED, not where it is defined.
    mock_get = mocker.patch("src.api_client.requests.get")
    
    # 2. Configure the mock response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_user_data
    
    # 3. Call the real code
    email = client.get_user_email(user_id=1)
    
    # 4. Assertions
    assert email == "test@example.com"
    mock_get.assert_called_once_with("https://api.example.com/users/1")
