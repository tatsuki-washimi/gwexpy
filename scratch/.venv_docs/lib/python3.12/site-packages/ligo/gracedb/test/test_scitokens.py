import os
import pytest
import scitokens
import tempfile
from unittest import mock

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
from datetime import datetime, timezone
from igwn_auth_utils.scitokens import IgwnAuthError

from ligo.gracedb.rest import GraceDb
from ligo.gracedb.token import check_token_expiration, find_token_or_none

TEST_ISSUER = "local"
TEST_AUDIENCE = "ANY"
TEST_SCOPE = "gracedb.read"


# -- fixtures ---------------

@pytest.fixture(scope="session")  # one per suite is fine
def private_key():
    return generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )


@pytest.fixture
def rtoken(private_key):
    """Create a token
    """
    # configure keycache
    from scitokens.utils import keycache
    kc = keycache.KeyCache.getinstance()
    kc.addkeyinfo(
        TEST_ISSUER,
        "test_key",
        private_key.public_key(),
        cache_timer=60,
    )

    # create token
    token = scitokens.SciToken(key=private_key, key_id="test_key")
    token.update_claims({
        "iss": TEST_ISSUER,
        "aud": TEST_AUDIENCE,
        "scope": TEST_SCOPE,
        "sub": 'albert.einstein@ligo.org',
    })
    serialized_token = token.serialize(issuer=TEST_ISSUER,
                                       lifetime=1000).decode("utf-8")
    return serialized_token


@pytest.fixture
def scitoken(private_key):
    """Create a token object
    """
    def _make_token(exp):
        # configure keycache
        from scitokens.utils import keycache
        kc = keycache.KeyCache.getinstance()
        kc.addkeyinfo(
            TEST_ISSUER,
            "test_key",
            private_key.public_key(),
            cache_timer=60,
        )

        # create token
        token = scitokens.SciToken(key=private_key, key_id="test_key")
        token.update_claims({
            "iss": TEST_ISSUER,
            "aud": TEST_AUDIENCE,
            "scope": TEST_SCOPE,
            "exp": exp,
            "sub": 'albert.einstein@ligo.org',
        })

        return token

    return _make_token


# -- test scitokens ---------

def test_scitokens(rtoken):
    with tempfile.TemporaryDirectory() as tmpdir:
        scitoken_file = os.path.join(tmpdir, "test_scitoken")

        with os.fdopen(os.open(scitoken_file,
                       os.O_RDWR | os.O_CREAT, 0o500), 'w+') as h:
            h.write(rtoken)

        # Initialize client
        with mock.patch.dict(os.environ, {'SCITOKEN_FILE': scitoken_file}):
            g = GraceDb()

        assert 'scitoken' in g.auth_type


def test_token_expiration(scitoken):
    # This strikes me as being kind of race-conditiony, but assign a
    # big buffer to make sure the token is within or outside
    # the validity period.
    valid_window = 1000
    now = int(datetime.now(timezone.utc).timestamp())

    # A valid token:
    assert check_token_expiration(scitoken(now + valid_window)) is False
    # An expired token:
    assert check_token_expiration(scitoken(now - valid_window)) is True
    # A valid token, but within the reload_buffer:
    assert check_token_expiration(scitoken(now - valid_window),
                                  reload_buffer=3600) is True


def test_finding_tokens(rtoken):
    host = "https://fakegracedb.ligo.org/"

    with tempfile.TemporaryDirectory() as tmpdir:
        scitoken_file = os.path.join(tmpdir, "test_scitoken")

        with os.fdopen(os.open(scitoken_file,
                       os.O_RDWR | os.O_CREAT, 0o500), 'w+') as h:
            h.write(rtoken)

        # Try finding the token:
        with mock.patch.dict(os.environ, {'SCITOKEN_FILE': scitoken_file}):
            assert isinstance(find_token_or_none(host),
                              scitokens.scitokens.SciToken)

        # temporarily remove any user's token if it exists, and then check
        # again:

        user_token_file = f'/tmp/bt_u{os.getuid()}'
        temp_token_file = '/tmp/.tmp_token'
        moved_file = os.path.exists(user_token_file)

        if moved_file:
            os.rename(user_token_file, temp_token_file)

        # Now check that the token can't be found:
        assert find_token_or_none(host) is None

        # Try again except raise an error:
        with pytest.raises(IgwnAuthError) as e:
            find_token_or_none(host, fail_if_noauth=True)

        assert "could not find a valid SciToken" in str(e.value)

        # Move the file back:
        if moved_file:
            os.rename(temp_token_file, user_token_file)


# This test is a WIP and will be improved in a future O4c release.
@pytest.mark.skip(reason="tested manually online, figure out this test")
def test_scitoken_reloading(rtoken):

    # Set up functions for tracking and mocking:
    set_up_conn_func = \
        'ligo.gracedb.adapter.GraceDbCredHTTPSConnection.connect'
    get_conn_func = \
        'requests.packages.urllib3.connectionpool.HTTPSConnectionPool'
    load_token_func = 'ligo.gracedb.token.find_token_or_none'
    token_expire_func = \
        'ligo.gracedb.token.check_token_expiration'

    # Set up the fake scitoken:
    with tempfile.TemporaryDirectory() as tmpdir:
        scitoken_file = os.path.join(tmpdir, "test_scitoken")

        with os.fdopen(os.open(scitoken_file,
                       os.O_RDWR | os.O_CREAT, 0o500), 'w+') as h:
            h.write(rtoken)

        with mock.patch(get_conn_func) as mock_get_conn, \
             mock.patch(set_up_conn_func) as mock_set_up_conn, \
             mock.patch(load_token_func) as mock_load_token, \
             mock.patch(token_expire_func) as mock_token_expire, \
             mock.patch.dict(os.environ, {'SCITOKEN_FILE': scitoken_file}):  # noqa: F841, E501

            # Now set up the client:
            g = GraceDb(reload_cred=True, reload_buffer=3600)

            # Confirm that it found the scitoken:
            assert 'scitoken' in g.auth_type

            # Make a call. Allow it to get past the typeerror to avoid
            # mocking the status-->response-->status_code
            g.get("https://fakeurl.com")

            # What should happen after this is, we should check that the
            # find_token_or_none, _expired_cred, check_token_expiration
            # functions were entered the expected number of times
            # right before the final request is mocked.
