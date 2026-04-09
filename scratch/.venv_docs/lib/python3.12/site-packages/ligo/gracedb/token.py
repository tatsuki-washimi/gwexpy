# -*- coding: utf-8 -*-
# Copyright (C) Alexander Pace, Duncan Meacher,
#               Duncan Macleod (2025)
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

# This file contains some helper tools and methods that aid with
# token renewal.

import datetime
import logging

from igwn_auth_utils.error import IgwnAuthError
from igwn_auth_utils.scitokens import find_token

log = logging.getLogger(__name__)

expired_message = ('token expiration ({exp}) -  current time ({utc}) '
                   'is less the than reload_buffer ({buf}). reloading token.')


def check_token_expiration(token, reload_buffer=0):
    """ Checks to see if a token is expiring within the window
        defined by reload_buffer. If no token is found at all, then
        return true, prompting the client to reload. """

    log.debug('Checking scitoken expiration.')

    if token is None:
        log.info('A valid token was not found. Reloading token '
                 'adaptor.')
        return True
    else:
        exp_time = datetime.datetime.utcfromtimestamp(token['exp'])
        now = datetime.datetime.utcnow()
        expired = (exp_time - now) <= \
            datetime.timedelta(seconds=reload_buffer)
        if expired:
            log.debug(expired_message.format(exp=exp_time,
                                             utc=now,
                                             buf=reload_buffer))
        else:
            log.debug('Token expiration is outside reload_buffer')

        return expired


def find_token_or_none(host, fail_if_noauth=False, scopes=None):
    """ Attempts to find a valid scitoken for the given host
        and scopes. If not, return None unless fail_if_noauth
        is true, then raise an error. """

    # Import the default token scope in the function to avoid
    # circular imports
    from .client import DEFAULT_TOKEN_SCOPE
    logging.debug('Finding a valid scitoken for new connection.')

    # If no scopes were provided, then use the default:
    scopes = DEFAULT_TOKEN_SCOPE if scopes is None else scopes

    # Find a valid token for the current audience (host) and scope.
    # If no token can be found and fail_if_noauth is True, then
    # raise an error.
    try:
        return find_token(host, scope=scopes)
    except IgwnAuthError as err:
        if not fail_if_noauth:
            return None
        else:
            raise err
