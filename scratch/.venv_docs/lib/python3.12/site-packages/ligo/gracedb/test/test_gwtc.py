import json
import pytest
from unittest import mock


def test_gwtc_create_incompat_server(safe_client, gwtc_smap_content):
    number = '4'

    # Set up links mock
    mock_link = mock.MagicMock()
    mock_link_dict = {'not-gwtc': mock_link}
    link_prop = 'ligo.gracedb.rest.GraceDb.links'

    with mock.patch('ligo.gracedb.rest.GraceDb.post'), \
         mock.patch(link_prop, mock_link_dict):  # noqa: F841
        with pytest.raises(NotImplementedError):
            safe_client.gwtc_create(gwtc_smap_content, number)


def test_gwtc_create_input_bad_smap(safe_client):
    # Mock service info dict
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'

    smap = 4
    number = '4'

    # Call:
    with mock.patch(si_property):
        with pytest.raises(ValueError):
            safe_client.gwtc_create(smap, number)


def test_gwtc_create_input_bad_number(safe_client, gwtc_smap_content):
    # Mock service info dict
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'

    smap = gwtc_smap_content
    number = 14

    # Call:
    with mock.patch(si_property):
        with pytest.raises(ValueError):
            safe_client.gwtc_create(smap, number)


def test_gwtc_create_input_with_smap_filename(safe_client, gwtc_smap_content):
    # Make up a filename:
    smap = 'smap.json'
    number = '4'
    smap_superevent = list(gwtc_smap_content)[0]

    # Set up templates mock
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'
    mock_template = mock.MagicMock()
    mock_template_dict = {'gwtc': mock_template}
    template_prop = 'ligo.gracedb.rest.GraceDb.templates'

    # Set up mock open
    open_func = 'ligo.gracedb.rest.open'
    mock_data = json.dumps(gwtc_smap_content)
    open_mocker = mock.mock_open(read_data=mock_data)

    with mock.patch('ligo.gracedb.rest.GraceDb.post') as mock_post, \
         mock.patch(open_func, open_mocker), \
         mock.patch(si_property), \
         mock.patch(template_prop, mock_template_dict): # noqa E127

        safe_client.gwtc_create(smap, number)

        post_call_args, post_call_kwargs = mock_post.call_args
        assert 'json' in post_call_kwargs
        assert smap_superevent in post_call_kwargs['json']['smap']


def test_gwtc_create_input_with_smap_json(safe_client, gwtc_smap_content):
    number = '4'
    smap_superevent = list(gwtc_smap_content)[0]

    # Set up templates mock
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'
    mock_template = mock.MagicMock()
    mock_template_dict = {'gwtc': mock_template}
    template_prop = 'ligo.gracedb.rest.GraceDb.templates'

    with mock.patch('ligo.gracedb.rest.GraceDb.post') as mock_post, \
         mock.patch(si_property), \
         mock.patch(template_prop, mock_template_dict):  # noqa: E127

        safe_client.gwtc_create(gwtc_smap_content, number)

        post_call_args, post_call_kwargs = mock_post.call_args
        assert 'json' in post_call_kwargs
        assert smap_superevent in post_call_kwargs['json']['smap']


def test_gwtc_get_incompat_server(safe_client):
    number = '4a'
    version = 1

    # Set up templates mock
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'
    mock_template = mock.MagicMock()
    mock_template_dict = {"not-gwtc-version-detail": mock_template}
    template_prop = 'ligo.gracedb.rest.GraceDb.templates'

    with mock.patch('ligo.gracedb.rest.GraceDb.get'), \
         mock.patch(si_property), \
         mock.patch(template_prop, mock_template_dict):  # noqa: F841
        with pytest.raises(NotImplementedError):
            safe_client.gwtc_get(number, version)


@pytest.mark.parametrize(
    "number,version",
    [
        (None, None),
        (4, None),
        ('4a', None),
        (None, '4'),
        ('4a', 'most_recent'),
        (4, 'latest'),
    ])
def test_gwtc_get_invalid_values(safe_client, number, version):
    # Set up templates mock
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'
    mock_template = mock.MagicMock()
    mock_template_dict = {"gwtc-version-detail": mock_template}
    template_prop = 'ligo.gracedb.rest.GraceDb.templates'

    with mock.patch('ligo.gracedb.rest.GraceDb.get'), \
         mock.patch(si_property), \
         mock.patch(template_prop, mock_template_dict):  # noqa: F841
        with pytest.raises(ValueError):
            safe_client.gwtc_get(number, version)


def test_gwtc_get(safe_client):
    number = '4a'
    version = 1

    # Set up templates mock
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'
    mock_template = mock.MagicMock()
    mock_template_dict = {"gwtc-version-detail": mock_template}
    template_prop = 'ligo.gracedb.rest.GraceDb.templates'

    with mock.patch('ligo.gracedb.rest.GraceDb.get') as mock_get, \
         mock.patch(si_property), \
         mock.patch(template_prop, mock_template_dict):  # noqa: E127
        safe_client.gwtc_get(number, version)

        get_call_args, get_call_kwargs = mock_get.call_args
        assert len(get_call_args) == 1
        assert get_call_kwargs == {}

# FIXME: mocking the generator object is non-zero effort and should
# be implemented in a future client release for gwtc_list


@pytest.mark.parametrize(
    "link_key", ['gwtc', 'not-gwtc']
)
def test_gwtc_numbers(safe_client, link_key):
    # Set up links mock
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'
    mock_link = mock.MagicMock()
    mock_link_dict = {link_key: mock_link}
    link_prop = 'ligo.gracedb.rest.GraceDb.links'

    with mock.patch('ligo.gracedb.rest.GraceDb.get') as mock_get, \
         mock.patch(si_property), \
         mock.patch(link_prop, mock_link_dict):  # noqa: F841

        if (link_key == 'gwtc'):
            safe_client.gwtc_numbers()
            get_call_args, get_call_kwargs = mock_get.call_args
            # no args to this method, but the get() passes the uri
            assert len(get_call_args) == 1
        else:
            with pytest.raises(NotImplementedError):
                safe_client.gwtc_numbers()


@pytest.mark.parametrize(
    "template_key,number", [
        ('gwtc-number-list', '4'),
        ('not-gwtc-number-list', '4'),
        ('gwtc-number-list', 4),
    ]
)
def test_gwtc_versions(safe_client, template_key, number):
    # Set up templates mock
    si_property = 'ligo.gracedb.rest.GraceDb.service_info'
    mock_template = mock.MagicMock()
    mock_template_dict = {template_key: mock_template}
    template_prop = 'ligo.gracedb.rest.GraceDb.templates'

    with mock.patch('ligo.gracedb.rest.GraceDb.get') as mock_get, \
         mock.patch(si_property), \
         mock.patch(template_prop, mock_template_dict):  # noqa: F841

        if (template_key == 'gwtc-number-list'):
            if isinstance(number, str):
                safe_client.gwtc_versions(number)
                get_call_args, get_call_kwargs = mock_get.call_args
                assert len(get_call_args) == 1
            else:
                with pytest.raises(ValueError):
                    safe_client.gwtc_versions(number)
        else:
            with pytest.raises(NotImplementedError):
                safe_client.gwtc_versions(number)
