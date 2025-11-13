#!/usr/bin/env python3
"""
Numeric-domain scanner with:
  • all padded / unpadded representations
  • 100-parallel requests
  • rich HTML stats (title, meta, JS/CSS, …)
  • Ctrl-C → immediate save of results collected so far
"""

import argparse
import csv
import json
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import socket
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------- #
# Global flag & handler for Ctrl-C
# --------------------------------------------------------------------------- #
INTERRUPT = False


def _signal_handler(sig, frame):
    global INTERRUPT
    if INTERRUPT:
        print("\nForced exit.", file=sys.stderr)
        sys.exit(1)
    INTERRUPT = True
    print("\nCtrl-C received – finishing current batch and saving results…", flush=True)


signal.signal(signal.SIGINT, _signal_handler)

# --------------------------------------------------------------------------- #
# Session & HTML parser
# --------------------------------------------------------------------------- #
def create_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; DomainScanner/1.0)"})
    return s


def _parse_html(html_bytes):
    """Return dict of HTML-derived stats (empty on non-HTML)."""
    try:
        soup = BeautifulSoup(html_bytes, "html.parser", from_encoding="utf-8")
    except Exception:
        return {}

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    meta_desc = None
    for m in soup.find_all("meta"):
        if m.get("name", "").lower() == "description":
            meta_desc = m.get("content")
            break

    visible_text = soup.get_text(separator=" ", strip=True)
    word_count = len(visible_text.split()) if visible_text else 0

    has_js = bool(soup.find_all("script"))
    has_css = bool(soup.find_all("link", rel="stylesheet") or soup.find_all("style"))

    num_links = len(soup.find_all("a", href=True))
    num_images = len(soup.find_all("img"))
    num_forms = len(soup.find_all("form"))

    tag_counter = Counter(tag.name for tag in soup.find_all())
    tag_counts = dict(tag_counter.most_common(10))

    lang = soup.html.get("lang") if soup.html else None

    return {
        "title": title,
        "meta_description": meta_desc,
        "has_js": has_js,
        "has_css": has_css,
        "num_links": num_links,
        "num_images": num_images,
        "num_forms": num_forms,
        "tag_counts": json.dumps(tag_counts) if tag_counts else None,
        "word_count": word_count,
        "lang": lang,
    }


def scrape_domain(num_str, session):
    url = f"http://{num_str}.com"
    try:
        resp = session.get(url, timeout=10, allow_redirects=True)
        if resp.status_code >= 400:
            return None

        final_url = resp.url
        size_bytes = len(resp.content)

        hostname = urlparse(final_url).hostname
        try:
            ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            ip = "Unknown"

        html_stats = {}
        if "text/html" in resp.headers.get("Content-Type", "").lower():
            html_stats = _parse_html(resp.content)

        result = {
            "numerical_url": num_str,
            "final_url": final_url,
            "size_bytes": size_bytes,
            "ip_address": ip,
        }
        result.update(html_stats)
        return result

    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException):
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Generate every representation (padded + unpadded)
# --------------------------------------------------------------------------- #
def generate_all_representations(max_num):
    if max_num < 0:
        return []
    max_digits = len(str(max_num))
    reps = set()
    for i in range(max_num + 1):
        reps.add(str(i))                     # without leading zeros
        reps.add(f"{i:0{max_digits}d}")      # with leading zeros
    return sorted(reps, key=lambda x: (len(x), x))


# --------------------------------------------------------------------------- #
# CSV writer (shared function – called on normal finish *or* interrupt)
# --------------------------------------------------------------------------- #
def write_csv(results, max_num):
    output_file = f"domain_scan_0_to_{max_num}_all_formats.csv"
    fieldnames = [
        "numerical_url", "final_url", "size_bytes", "ip_address",
        "title", "meta_description", "has_js", "has_css",
        "num_links", "num_images", "num_forms",
        "tag_counts", "word_count", "lang",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return output_file


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main(max_num):
    global INTERRUPT

    numbers = generate_all_representations(max_num)
    total = len(numbers)
    if total == 0:
        print("Nothing to scan.")
        return

    results = []
    session = create_session()

    print(f"Scanning {total} domain variants (0.com → {max_num}.com, all paddings)…")
    print(f"Examples: {', '.join(numbers[:5])}, …, {', '.join(numbers[-3:])}")
    print("Progress: 0/", total, end="", flush=True)

    start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=500) as executor:
        future_to_num = {executor.submit(scrape_domain, n, session): n for n in numbers}

        for future in as_completed(future_to_num):
            # Stop early if user pressed Ctrl-C
            if INTERRUPT:
                break

            result = future.result()
            completed += 1

            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"\rProgress: {completed}/{total} ({rate:.1f}/s)", end="", flush=True)

            if result:
                results.append(result)

    # ------------------------------------------------------------------- #
    # Finalise (normal finish or interrupt)
    # ------------------------------------------------------------------- #
    print(f"\n\nScan {'interrupted' if INTERRUPT else 'finished'}. "
          f"Found {len(results)} live domains.")
    
    results.sort(key=lambda e: e['numerical_url'])
    csv_file = write_csv(results, max_num)
    print(f"Results saved to → {csv_file}")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape numeric .com domains with/without leading zeros."
    )
    parser.add_argument(
        "max_num",
        type=int,
        help="Maximum number (e.g. 11 → 0-11 + 00-09)"
    )
    args = parser.parse_args()

    if args.max_num < 0:
        print("Error: max_num must be >= 0", file=sys.stderr)
        sys.exit(1)

    main(args.max_num)