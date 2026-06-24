"""
whois_lookup.py
---------------
Fetches WHOIS / domain registration information for a given hostname.
Returns domain age, expiry length, and registrar details.

Requires: python-whois  (pip install python-whois)
"""

from __future__ import annotations

import datetime
from typing import Optional
import urllib.parse


def _safe_import_whois():
    try:
        import whois  # python-whois package
        return whois
    except ImportError:
        return None


def get_domain_info(url: str) -> dict:
    """
    Return domain registration details for the given URL.

    Parameters
    ----------
    url : str
        Full URL or bare hostname.

    Returns
    -------
    dict with keys:
        domain              – extracted hostname
        creation_date       – datetime or None
        expiration_date     – datetime or None
        registrar           – str or None
        domain_age_days     – int or -1 if unknown
        domain_expiry_days  – int or -1 if unknown
        dns_record_exists   – bool
        whois_available     – bool
    """
    info = {
        "domain":             None,
        "creation_date":      None,
        "expiration_date":    None,
        "registrar":          None,
        "domain_age_days":    -1,
        "domain_expiry_days": -1,
        "dns_record_exists":  False,
        "whois_available":    False,
    }

    # --- Extract hostname ----------------------------------------------------
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        hostname = urllib.parse.urlparse(url).hostname or ""
        # strip leading www.
        if hostname.startswith("www."):
            hostname = hostname[4:]
        info["domain"] = hostname
    except Exception:
        return info

    # --- DNS check (lightweight socket) --------------------------------------
    try:
        import socket
        socket.setdefaulttimeout(3)
        socket.gethostbyname(hostname)
        info["dns_record_exists"] = True
    except Exception:
        info["dns_record_exists"] = False

    # --- WHOIS ---------------------------------------------------------------
    whois = _safe_import_whois()
    if whois is None:
        return info  # library not installed – return partial info

    try:
        w = whois.whois(hostname)
        info["whois_available"] = True

        # creation date
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        if isinstance(cd, datetime.datetime):
            info["creation_date"] = cd
            age = (datetime.datetime.utcnow() - cd).days
            info["domain_age_days"] = max(age, 0)

        # expiration date
        ed = w.expiration_date
        if isinstance(ed, list):
            ed = ed[0]
        if isinstance(ed, datetime.datetime):
            info["expiration_date"] = ed
            expiry = (ed - datetime.datetime.utcnow()).days
            info["domain_expiry_days"] = max(expiry, 0)

        # registrar
        info["registrar"] = w.registrar or "Unknown"

    except Exception:
        pass  # WHOIS lookup failed – leave defaults

    return info
