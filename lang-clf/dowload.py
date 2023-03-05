import argparse
import requests
from datetime import datetime
import os
from urllib3 import PoolManager
import json

def download(lang_file, out_dir, no_samples):
    print(f"language file: {lang_file}")
    print(f"samples: {no_samples}")
    langs = []

    http = PoolManager(maxsize=50, block=True)

    with open(lang_file, 'r') as f:
        lines = f.read().splitlines()
        print(f"checking {len(lines)} WIKIPEDIA subdomains")
        for line in lines:
            # use first column as label
            try:
                lang = line.split(';')[0].strip()
                print(f"checking {lang}")
                url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
                resp = http.request('GET', url)
                data = resp.data.decode('utf-8')
                values = json.loads(data)
                if 'extract' in values:
                    langs.append(lang)
                else:
                    print(f"ignoring language {lang} -> extract not found in response!")
            except requests.exceptions.ConnectionError:
                print(f"ignoring language {lang} -> WIKIPEDIA subdomain not found!")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d")
    min_len = 30

    for i, lang in enumerate(langs):
        print(f"loading language data: {lang} ({i + 1}/{len(langs)})")
        samples = []
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
        for i in range(0, no_samples):
            resp = http.request('GET', url)
            if resp.status == 200:
                data = resp.data.decode('utf-8')
                values = json.loads(data)
                if 'extract' in values:
                    txt = values['extract']
                    txt = txt.strip().replace('\n', ' ')
                    if len(txt) >= min_len:
                        samples.append(txt)
                    else:
                        print(f"ignore txt sample {lang} ({i+1}/{no_samples}) -> len {len(txt)} < min ({min_len})")
            else:
                print(f"Could not load txt sample ${lang} ({i+1}/{len(no_samples)}) -> code {resp.status}")

        filename = os.path.join(out_dir, lang, f"{lang}_{timestamp}.txt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'a') as f:
            for txt in samples:
                f.write(txt + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download language samples.')
    parser.add_argument("--out", type=str, default='data')
    parser.add_argument("--file", type=str, default='iso-639-1.csv')
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    download(args.file, args.out, args.samples)
