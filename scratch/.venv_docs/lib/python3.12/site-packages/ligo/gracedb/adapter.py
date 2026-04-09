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
# along with gracedb.  If not, see <http://www.gnu.org/licenses/>

# Sources:
#  1) https://stackoverflow.com/questions/45539422/can-we-reload-a-page-url-
#     in-python-using-urllib-or-urllib2-or-requests-or-mechan

#  2) https://2.python-requests.org/en/master/user/advanced/#example-
#     specific-ssl-version

#  3) https://urllib3.readthedocs.io/en/1.2.1/pools.html

import logging

from functools import partial
from igwn_auth_utils import scitoken_authorization_header
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.connection import HTTPSConnection
from requests.packages.urllib3.connectionpool \
    import HTTPSConnectionPool, HTTPConnectionPool
from .cert import check_certificate_expiration
from .token import check_token_expiration, find_token_or_none

# Some variables:
auth_header = 'Authorization'

# initiate logging:
log = logging.getLogger(__name__)


class GraceDbCredAdapter(HTTPAdapter):
    def __init__(self, cert=None,
                 token=None, reload_buffer=0,
                 fail_if_noauth=False, **kwargs):
        super(GraceDbCredAdapter, self).__init__(**kwargs)
        https_pool_cls = partial(
            GraceDbCredHTTPSConnectionPool,
            cert=cert,
            token=token,
            reload_buffer=reload_buffer,
            fail_if_noauth=fail_if_noauth)

        self.poolmanager.pool_classes_by_scheme = {
            'http': HTTPConnectionPool,
            'https': https_pool_cls
        }
        self.poolmanager.connection_pool_kw['maxsize'] = 10


class GraceDbCredHTTPSConnection(HTTPSConnection):
    def __init__(self, host, cert=None,
                 token=None, reload_buffer=0, **kwargs):
        # At this point, te HTTPSConnection is initialized
        # but unconnected. Set this property to 'True'
        self.unestablished_connection = True
        super(GraceDbCredHTTPSConnection, self).__init__(host, **kwargs)

    @property
    def unestablished_connection(self):
        return self._unestablished_connection

    @unestablished_connection.setter
    def unestablished_connection(self, value):
        self._unestablished_connection = value

    def connect(self):
        # Connected. After this step, the unestablished
        # property is false.
        self.unestablished_connection = False
        super(GraceDbCredHTTPSConnection, self).connect()


class GraceDbCredHTTPSConnectionPool(HTTPSConnectionPool):
    # ConnectionPool object gets used in the HTTPAdapter.
    # "ConnectionCls" is a HTTP(S)COnnection object to use
    # As the underlying connection.

    # Source: https://urllib3.readthedocs.io/en/latest/
    #         reference/#module-urllib3.connectionpool

    ConnectionCls = GraceDbCredHTTPSConnection

    def __init__(self, host, port=None, cert=None,
                 token=None, reload_buffer=0,
                 fail_if_noauth=False, **kwargs):

        super(GraceDbCredHTTPSConnectionPool, self).__init__(
            host, port=port, **kwargs)

        self._cert = cert
        # self._token is a proxy for determining if this is a token auth
        # session and self._cur_token is the current token to validate
        # and refresh
        self._token = bool(token)
        self._cur_token = token
        self._reload_buffer = reload_buffer
        self._fail_if_noauth = fail_if_noauth

    def _expired_cred(self):
        if self._token:
            return check_token_expiration(
                self._cur_token,
                self._reload_buffer)
        elif self._cert:
            return check_certificate_expiration(
                self._cert,
                self._reload_buffer)
        else:
            raise ValueError('No renewable credentials (token or '
                             'certificate) provided.')

    def _get_conn(self, timeout=None):
        while True:
            # Start the connection object. At this step, the connection
            # unestablished variable is true
            conn = super(GraceDbCredHTTPSConnectionPool, self)._get_conn(
                timeout)

            # 'returning' the connection object then triggers the
            # connection to be established. Establish a new connection
            # if it's unestablished, or if the cert expiration is within the
            # reload buffer. Establishing the new connection will (hopefully
            # load the new cert.
            if conn.unestablished_connection or not self._expired_cred():
                return conn
            else:
                if self._token:

                    # Attempt to get a new token. If no valid token is found,
                    # then None, and the adaptor will check for a valid token
                    # again on each subsequent request. if fail_if_noauth=true,
                    # then raise an error.
                    self._cur_token = find_token_or_none(
                        self.host,
                        fail_if_noauth=self._fail_if_noauth)

                    # Now update the headers. If there is no current token,
                    # pop the auth header to reflect the lack of a valid token:
                    if self._cur_token is None:
                        self.headers.pop(auth_header, None)
                    # Otherwise, update the auth header with the new token:
                    else:
                        log.info('Reconstructing token header for scitoken.')
                        self.headers.update({
                            auth_header:
                                scitoken_authorization_header(self._cur_token)
                        })

            # otherwise, kill the connection which will reset unestablished_..
            # to true and then exit the loop.
                conn.close()
