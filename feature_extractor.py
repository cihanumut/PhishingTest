import re
from urllib.parse import urlparse
import socket
import numpy as np
import ssl
import whois
from datetime import datetime
import time

class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url', 'qty_questionmark_url',
            'qty_equal_url', 'qty_at_url', 'qty_and_url', 'qty_exclamation_url', 'qty_space_url',
            'qty_tilde_url', 'qty_comma_url', 'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url',
            'qty_dollar_url', 'qty_percent_url', 'qty_tld_url', 'length_url', 'qty_dot_domain',
            'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain', 'qty_questionmark_domain',
            'qty_equal_domain', 'qty_at_domain', 'qty_and_domain', 'qty_exclamation_domain',
            'qty_space_domain', 'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain',
            'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain', 'qty_percent_domain',
            'qty_vowels_domain', 'domain_length', 'domain_in_ip', 'server_client_domain',
            'qty_dot_directory', 'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
            'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory', 'qty_and_directory',
            'qty_exclamation_directory', 'qty_space_directory', 'qty_tilde_directory', 'qty_comma_directory',
            'qty_plus_directory', 'qty_asterisk_directory', 'qty_hashtag_directory', 'qty_dollar_directory',
            'qty_percent_directory', 'directory_length', 'qty_dot_file', 'qty_hyphen_file',
            'qty_underline_file', 'qty_slash_file', 'qty_questionmark_file', 'qty_equal_file',
            'qty_at_file', 'qty_and_file', 'qty_exclamation_file', 'qty_space_file', 'qty_tilde_file',
            'qty_comma_file', 'qty_plus_file', 'qty_asterisk_file', 'qty_hashtag_file', 'qty_dollar_file',
            'qty_percent_file', 'file_length', 'qty_dot_params', 'qty_hyphen_params', 'qty_underline_params',
            'qty_slash_params', 'qty_questionmark_params', 'qty_equal_params', 'qty_at_params',
            'qty_and_params', 'qty_exclamation_params', 'qty_space_params', 'qty_tilde_params',
            'qty_comma_params', 'qty_plus_params', 'qty_asterisk_params', 'qty_hashtag_params',
            'qty_dollar_params', 'qty_percent_params', 'params_length', 'tld_present_params',
            'qty_params', 'email_in_url', 'time_response', 'domain_spf', 'asn_ip',
            'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved', 'qty_nameservers',
            'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate', 'qty_redirects',
            'url_google_index', 'domain_google_index', 'url_shortened'
        ]
        
        # Shared DNS resolver using Google Public DNS for reliability
        import dns.resolver
        self._resolver = dns.resolver.Resolver()
        self._resolver.nameservers = ['8.8.8.8', '8.8.4.4']
        self._resolver.timeout = 10
        self._resolver.lifetime = 10

    # ── Network probes ──────────────────────────────────────────────

    def _check_ssl(self, domain):
        """Returns (has_ssl: 0|1, response_time: float)"""
        start = time.time()
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=3) as sock:
                with ctx.wrap_socket(sock, server_hostname=domain):
                    return 1, round(time.time() - start, 2)
        except Exception:
            return 0, round(time.time() - start, 2)

    def _dns_resolve(self, domain):
        """Returns (ip_count, has_dns: bool)"""
        try:
            ips = socket.gethostbyname_ex(domain)[2]
            return len(ips), True
        except Exception:
            return 0, False

    def _dns_ns_count(self, domain):
        """Returns nameserver count via DNS NS query."""
        try:
            ans = self._resolver.resolve(domain, 'NS')
            return len(ans)
        except Exception:
            return 0

    def _dns_mx_count(self, domain):
        """Returns MX record count."""
        try:
            ans = self._resolver.resolve(domain, 'MX')
            return len(ans)
        except Exception:
            return 0

    def _dns_ttl(self, domain):
        """Returns TTL of A record."""
        try:
            ans = self._resolver.resolve(domain, 'A')
            return ans.rrset.ttl
        except Exception:
            return np.nan

    def _check_spf(self, domain):
        """Check if domain has SPF record. Returns 0 or 1."""
        try:
            answers = self._resolver.resolve(domain, 'TXT')
            for rdata in answers:
                if 'v=spf1' in str(rdata):
                    return 1
        except Exception:
            pass
        return 0

    def _get_asn(self, domain):
        """Get ASN number for the domain's IP via Cymru DNS."""
        try:
            ip = socket.gethostbyname(domain)
            parts = ip.split('.')
            parts.reverse()
            query = '.'.join(parts) + '.origin.asn.cymru.com'
            ans = self._resolver.resolve(query, 'TXT')
            for rdata in ans:
                txt = str(rdata).strip('"')
                return int(txt.split('|')[0].strip())
        except Exception:
            pass
        return np.nan

    def _get_whois(self, domain):
        """Returns (activation_days, expiration_days) or (np.nan, np.nan)."""
        try:
            w = whois.whois(domain)
            now = datetime.now()

            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if creation and hasattr(creation, 'tzinfo') and creation.tzinfo:
                creation = creation.replace(tzinfo=None)
            activation = (now - creation).days if creation else np.nan

            expiry = w.expiration_date
            if isinstance(expiry, list):
                expiry = expiry[0]
            if expiry and hasattr(expiry, 'tzinfo') and expiry.tzinfo:
                expiry = expiry.replace(tzinfo=None)
            expiration = (expiry - now).days if expiry else np.nan

            return activation, expiration
        except Exception as e:
            print(f"  [WHOIS] {domain}: {e}")
            return np.nan, np.nan

    # ── Main extraction ─────────────────────────────────────────────

    def extract_features(self, url):
        original_url = url
        parsed = urlparse(url)
        if not parsed.scheme:
            url = 'http://' + url
            parsed = urlparse(url)

        domain = parsed.netloc
        if ':' in domain:
            domain = domain.split(':')[0]
        path   = parsed.path
        params = parsed.query

        # ── Character counts ────────────────────────────────────────
        char_map = {
            '.': 'dot', '-': 'hyphen', '_': 'underline', '/': 'slash',
            '?': 'questionmark', '=': 'equal', '@': 'at', '&': 'and',
            '!': 'exclamation', ' ': 'space', '~': 'tilde', ',': 'comma',
            '+': 'plus', '*': 'asterisk', '#': 'hashtag', '$': 'dollar',
            '%': 'percent'
        }

        directory = '/'.join(path.split('/')[:-1])
        file_name = path.split('/')[-1]

        features = {}
        for char, name in char_map.items():
            features[f"qty_{name}_url"]       = original_url.count(char)
            features[f"qty_{name}_domain"]    = domain.count(char)
            features[f"qty_{name}_directory"] = directory.count(char)
            features[f"qty_{name}_file"]      = file_name.count(char)
            # Dataset uses -1 (→ NaN) when there are no params
            features[f"qty_{name}_params"]    = params.count(char) if params else np.nan

        # ── Structural features ─────────────────────────────────────
        features['qty_tld_url']          = len(re.findall(r'\.[a-z]+', original_url))
        features['length_url']           = len(original_url)
        features['qty_vowels_domain']    = len(re.findall(r'[aeiou]', domain, re.I))
        features['domain_length']        = len(domain)
        features['domain_in_ip']         = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain) else 0
        features['server_client_domain'] = 1 if ('server' in domain or 'client' in domain) else 0
        features['directory_length']     = len(directory)
        features['file_length']          = len(file_name)

        if params:
            features['params_length']       = len(params)
            features['tld_present_params']  = 1 if any(t in params for t in ['.com','.net','.org']) else 0
            features['qty_params']          = len(params.split('&'))
        else:
            features['params_length']       = np.nan
            features['tld_present_params']  = np.nan
            features['qty_params']          = np.nan

        # ── Network probes (real data) ──────────────────────────────
        ip_count, has_dns = self._dns_resolve(domain)
        has_ssl, resp_time = self._check_ssl(domain)
        ns_count = self._dns_ns_count(domain)
        mx_count = self._dns_mx_count(domain)
        ttl_val  = self._dns_ttl(domain)
        spf      = self._check_spf(domain)
        asn      = self._get_asn(domain)
        activation, expiration = self._get_whois(domain)

        features['email_in_url']           = 1 if re.search(r'[\w.-]+@[\w.-]+', original_url) else 0
        features['time_response']          = round(resp_time * 2) / 2  # Bucket to 0.5s intervals (0, 0.5, 1.0, ...)
        features['domain_spf']             = spf
        features['asn_ip']                 = asn
        features['time_domain_activation'] = activation
        features['time_domain_expiration'] = expiration
        features['qty_ip_resolved']        = ip_count if ip_count > 0 else np.nan
        features['qty_nameservers']        = ns_count if ns_count > 0 else np.nan
        features['qty_mx_servers']         = mx_count  # 0 is valid (some legit domains have 0)
        features['ttl_hostname']           = round(ttl_val / 1000) * 1000 if ttl_val and not np.isnan(ttl_val) else np.nan  # Round to nearest 1000
        features['tls_ssl_certificate']    = has_ssl
        features['qty_redirects']          = 0    # Can't measure without full HTTP chain
        features['url_google_index']       = 0    # Can't measure without Google API
        features['domain_google_index']    = 0    # Can't measure without Google API

        # URL shortener detection
        shorteners = ['bit.ly','tinyurl.com','goo.gl','t.co','ow.ly','is.gd',
                       'buff.ly','adf.ly','bit.do','shorte.st','tiny.cc','rb.gy','cutt.ly']
        features['url_shortened'] = 1 if domain in shorteners else 0

        # ── Build vector ────────────────────────────────────────────
        vector = [features.get(name, np.nan) for name in self.feature_names]
        return np.array(vector, dtype=np.float64).reshape(1, -1)
