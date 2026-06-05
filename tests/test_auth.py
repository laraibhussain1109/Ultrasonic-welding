from blower_inspection.auth import AuthStore, hash_password, verify_password


def test_hash_and_verify_password():
    encoded = hash_password("secret", salt=b"fixed-salt")
    assert verify_password("secret", encoded)
    assert not verify_password("wrong", encoded)


def test_auth_store_roles(tmp_path):
    path = tmp_path / "users.json"
    store = AuthStore(path)
    store.upsert_user("admin", "pw", "admin")
    user = store.authenticate("admin", "pw")
    assert user is not None
    assert user.is_admin
    assert store.authenticate("admin", "bad") is None
