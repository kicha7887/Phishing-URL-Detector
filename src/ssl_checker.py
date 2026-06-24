"""
ssl_checker.py
--------------
Verifies the SSL/TLS certificate of a given hostname.
Returns validity status, issuer, expiry date, and days remaining.
"""

from __future__ import annotations

import ssl
import socket
import datetime
from typing import Optional
import urllib.parse


def check_ssl(url: str, timeout: int = 5) -> dict:
    """
    Check SSL certificate for the host in *url*.

    Parameters
    ----------
    url : str
        Full URL or bare hostname.
    timeout : int
        Socket timeout in seconds.

    Returns
    -------
    dict with keys:
        ssl_valid           – bool
        ssl_expired         – bool
        ssl_issuer          – str or None
        ssl_expiry_date     – datetime or None
        ssl_days_remaining  – int or -1
        ssl_error           – str or None  (set on failure)
    """
    result = {
        "ssl_valid":          False,
        "ssl_expired":        False,
        "ssl_issuer":         None,
        "ssl_expiry_date":    None,
        "ssl_days_remaining": -1,
        "ssl_error":          None,
    }

    # --- Extract hostname ----------------------------------------------------
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        hostname = urllib.parse.urlparse(url).hostname or ""
    except Exception as exc:
        result["ssl_error"] = str(exc)
        return result

    # --- Connect and fetch certificate ---------------------------------------
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        # Parse expiry
        not_after_str = cert.get("notAfter", "")
        if not_after_str:
            not_after = datetime.datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z")
            result["ssl_expiry_date"] = not_after
            days_left = (not_after - datetime.datetime.utcnow()).days
            result["ssl_days_remaining"] = days_left
            result["ssl_expired"] = days_left < 0

        # Parse issuer
        issuer_dict = dict(x[0] for x in cert.get("issuer", []))
        result["ssl_issuer"] = issuer_dict.get("organizationName", "Unknown")

        result["ssl_valid"] = not result["ssl_expired"]

    except ssl.SSLCertVerificationError as exc:
        result["ssl_error"] = f"Certificate verification failed: {exc}"
    except ssl.SSLError as exc:
        result["ssl_error"] = f"SSL error: {exc}"
    except (socket.timeout, ConnectionRefusedError, OSError) as exc:
        result["ssl_error"] = f"Connection error: {exc}"
    except Exception as exc:
        result["ssl_error"] = f"Unknown error: {exc}"

    return result
