# -*- coding: utf-8 -*-
# Copyright (C) Alexander Pace, Tanner Prestegard,
#               Branson Stephens, Brian Moe (2020)
#
# This file is part of gracedb
#
# gracedb is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# It is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gracedb.  If not, see <http://www.gnu.org/licenses/>.
import json as json_lib
import warnings
from datetime import datetime
from pprint import pprint
from requests.adapters import HTTPAdapter, Retry
from requests import codes
from urllib.parse import urlparse

from .version import __version__
from .adapter import GraceDbCredAdapter
from .utils import hook_response, raise_status_exception

from igwn_auth_utils.requests import (HTTPSciTokenAuth, Session)

DEFAULT_SERVICE_URL = "https://gracedb.ligo.org/api/"
DEFAULT_RETRY_CODES = [
    codes.BAD_GATEWAY,      # 502
    codes.GATEWAY_TIMEOUT,  # 504
    codes.REQUEST_TIMEOUT,  # 408
    codes.CONFLICT,         # 409
]

DEFAULT_ALLOWED_METHODS = None
DEFAULT_TOKEN_SCOPE = "gracedb.read"


class GraceDBClient(Session):
    """
    url (:obj:`str`, optional): URL of server API
    cred (:obj:`tuple` or :obj:`str, optional): a tuple or list of
        (``/path/to/cert/file``, ``/path/to/key/file) or a single path to
        a combined proxy file (if using an X.509 certificate for
        authentication)
    force_noauth (:obj:`bool`, optional): set to True if you want to skip
        credential lookup and use this client as an unauthenticated user
    fail_if_noauth (:obj:`bool`, optional): set to True if you want the
        constructor to fail if no authentication credentials are provided
        or found, or if no valid scitoken credentials were found when
        using the ``reload_cred`` option
    reload_certificate (:obj:`bool`, optional): if ``True``, your
        certificate will be checked before each request whether it is
        within ``reload_buffer`` seconds of expiration, and if so, it will
        be reloaded. Useful for processes which may live longer than the
        certificate lifetime and have an automated method for certificate
        renewal. The path to the new/renewed certificate **must** be the
        same as for the old certificate. To be deprecated in favor of
        ``reload_cred``.
    reload_cred (:obj:`bool`, optional): if ``True``, the user's token
        or certificate will be checked before each request if it is within
        ``reload_buffer`` seconds of expiration, and if so, will be
        reloaded. Used for long-lived processes in conjunction with a
        user's own credential reloading mechanism.
    reload_buffer (:obj:`int`, optional): buffer (in seconds) for reloading
        a certificate in advance of its expiration. Only used if
        ``reload_certificate`` is ``True``.
    retries (:obj:`int`, optional): the maximum number of retries to
        attempt on an error from the server. Default 5.
    backoff_factor (:obj:`float`, optional): The backoff factor for retrying.
        Default: 0.1. Refer to urllib3 documentation for usage.
    retry_codes (:obj:`list`, optional): List of HTTPError codes to retry
        on. Default: [502, 504, 408, 409].
    allowed_methods (:obj:`list` or None, optional): whitelist of allowed
        http verbs to retry. Default: None (all verbs).
    use_auth: (:obj:`str`, optional): Specify the authentication method
        to be used. Choices: ``all``, ``scitoken``, ``x509``. Default:
        ``all``.

    Authentication details:

    By default, the code will search for available scitokens, and then X.509
    certificates. The credentials will then be used in that order of
    preference unless overridden by the ``use_auth`` variable.

    To use an X.509 certificate:
        Provide a path to an X.509 certificate and key or a single
           combined proxy file using the ``cred`` parameter.
    Or:
        The code will look for a certificate in a default location
            (``/tmp/x509up_u%d``, where ``%d`` is your user ID)
    """

    def __init__(self, url=DEFAULT_SERVICE_URL, cred=None,
                 reload_certificate=False,
                 reload_cred=False, reload_buffer=300,
                 use_auth='all', retries=5,
                 retry_codes=DEFAULT_RETRY_CODES,
                 allowed_methods=DEFAULT_ALLOWED_METHODS,
                 backoff_factor=0.1, *args, **kwargs):

        # Set which auth method to use if specified, otherwise use all.
        if use_auth == 'x509':
            if cred is None:
                cred = True
            kwargs.setdefault("token", False)
        elif use_auth == 'scitoken':
            kwargs.setdefault("token", True)
            if cred is None:
                cred = False
        kwargs.setdefault("token_scope", DEFAULT_TOKEN_SCOPE)

        super().__init__(cert=cred, url=url, *args, **kwargs)

        # Initialize variables:
        self.host = urlparse(url).hostname
        self._fail_if_noauth = kwargs.get('fail_if_noauth')

        # Define auth_type based on.... type of auth.
        self.auth_type = set()
        if use_auth != "x509" and isinstance(self.auth, HTTPSciTokenAuth):
            self.auth_type.add("scitoken")
        if use_auth != 'scitoken' and self.cert:
            self.auth_type.add("x509")

        # Update session headers:
        self.headers.update(self._update_headers())

        # Adjust the response via a session hook:
        self.hooks = {'response': [hook_response, raise_status_exception]}

        # Add the retrying adaptor:
        # https://stackoverflow.com/a/35504626
        # Sanity check the inputs:

        # 'retries' must be a positive (or zero) integer:
        if (not isinstance(retries, int) or retries < 0):
            raise ValueError('Invalid value of retries')

        # 'retry_codes' must be a list:
        if not isinstance(retry_codes, list):
            raise ValueError('retry_codes must be a list')

        if retries > 0:
            # the allowed_methods option was added as of urllib3==1.26.1
            # and replaces method_whitelist:
            if hasattr(Retry.DEFAULT, 'allowed_methods'):
                retries = Retry(total=retries,
                                backoff_factor=backoff_factor,
                                status_forcelist=retry_codes,
                                allowed_methods=allowed_methods)
            else:
                retries = Retry(total=retries,
                                backoff_factor=backoff_factor,
                                status_forcelist=retry_codes,
                                method_whitelist=allowed_methods)

        if reload_cred and isinstance(self.auth, HTTPSciTokenAuth) and \
                use_auth != 'x509':
            # Get the token object to pass to the reloading adapter.
            # Fail with an error if a token cannot be found 
            token = self.auth.find_token(self.host, error=True)
            self.mount('https://', GraceDbCredAdapter(
                       token=token,
                       fail_if_noauth=self._fail_if_noauth,
                       reload_buffer=reload_buffer,
                       max_retries=retries))
        elif (reload_certificate or reload_cred) and self.cert:
            if reload_certificate:
                warning_text = ('reload_certificate has been deprecated in '
                                'favor of reload_cred and will be removed in '
                                'a future release.')
                warnings.warn(warning_text,
                              DeprecationWarning,
                              stacklevel=2)
            self.mount('https://', GraceDbCredAdapter(
                       cert=self.cert,
                       reload_buffer=reload_buffer,
                       max_retries=retries))
        else:
            self.mount('https://', HTTPAdapter(max_retries=retries))

    def _update_headers(self):
        """ Update the sessions' headers:
        """
        new_headers = {}
        # Assign the user agent. This shows up in gracedb's log files.
        new_headers.update({'User-Agent':
                            'gracedb-client/{}'.format(__version__)})
        new_headers.update({'Accept-Encoding':
                            'gzip, deflate'})
        return new_headers

    # hijack 'Session.request':
    # https://2.python-requests.org/en/master/api/#requests.Session.request
    def request(
            self, method, url, params=None, data=None, headers=None,
            cookies=None, files=None, auth=None, timeout=None,
            allow_redirects=True, proxies=None, hooks=None, stream=None,
            verify=None, cert=None, json=None):
        return super().request(
            method, url, params=params, data=data, headers=headers,
            cookies=cookies, files=files, auth=auth,
            timeout=timeout, allow_redirects=True, proxies=proxies,
            hooks=hooks, stream=stream, verify=verify, cert=cert, json=json)

    # Extra definitions to return closed contexts' connections
    # back to the pool:
    # https://stackoverflow.com/questions/48160728/resourcewarning
    # -unclosed-socket-in-python-3-unit-test
    def close(self):
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # For getting files, return a "raw" file-type object.
    # Automatically decode content.
    def get_file(self, url, **kwargs):
        resp = self.get(url, stream=True, **kwargs)
        resp.raw.decode_content = True
        resp.raw.status_code = resp.status_code
        resp.raw.json = resp.json
        return resp.raw

    # Return client credentials
    def show_credentials(self, print_output=True):
        """ Prints authentication type and credentials info."""
        output = {}
        if not self.auth_type:
            output = {'auth_type(s)': 'No auth type found'}

        if 'scitoken' in self.auth_type:
            token = self.auth.find_token(self.host, error=False)
            if token is not None:
                exp = datetime.utcfromtimestamp(token['exp']).isoformat()
                output['scitoken'] = {
                    'SciToken subject': token['sub'],
                    'SciToken expiration': exp,
                    'SciToken scope': token['scope'],
                    'SciToken audience': token['aud'],
                }
        if 'x509' in self.auth_type:
            if isinstance(self.cert, tuple):
                cert = {'cert_file': self.cert[0],
                        'key_file': self.cert[1]}
            elif isinstance(self.cert, str):
                cert = {'cert_file': self.cert,
                        'key_file': self.cert}
            else:
                raise ValueError("Problem reading authentication certificate")
            output['x509'] = cert

        if print_output:
            pprint(output)
        else:
            return output

    def get_user_info(self):
        """Get information from the server about your user account."""
        user_info_link = self.links.get('user-info', None)
        if user_info_link is None:
            raise RuntimeError('Server does not provide a user info endpoint')
        return self.get(user_info_link)

    @classmethod
    def load_json_from_response(cls, response):
        """ Always return a json content response, even when the server
            provides a 204: no content"""
        # Check if response exists:
        if not response:
            raise ValueError("No response object provided")

        # Check if there is response content. If not, create it.
        if response.content == 'No Content':
            response_content = '{}'

        # Some responses send back strings of strings. This iterates
        # until proper dict is returned, or if it doesn't make progress.
        num_tries = 1
        response_content = response.content.decode('utf-8')

        while isinstance(response_content, str) and num_tries < 3:
            response_content = json_lib.loads(response_content)
            num_tries += 1

        if isinstance(response_content, dict):
            return response_content
        else:
            return ValueError("ERROR: got unexpected content from "
                              "the server: {}".format(response_content))
