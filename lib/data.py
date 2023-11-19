import argparse
from datetime import datetime
import os
from urllib3 import PoolManager, Retry
import json

from urllib3.exceptions import MaxRetryError


def get_languages_from_file(lang_file):
    with open(lang_file, 'r') as f:
        languages = []
        lines = f.read().splitlines()
        for line in lines:
            languages.append(line.split(';')[0].strip())
        return languages


def download(lang_file, out_dir, no_samples):
    print(f"language file: {lang_file}")
    print(f"samples: {no_samples}")
    langs = []

    retries = Retry(connect=5, read=2, redirect=5, status=2)
    http = PoolManager(maxsize=50, block=True, retries=retries)



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
            except MaxRetryError as ex:
                print(f"ignoring language {lang} -> {ex.reason}")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d")

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
                    samples.append(txt)
            else:
                print(f"Could not load txt sample {lang} ({i+1}/{no_samples}) "
                      f"-> http error {resp.status} ({resp.reason})")

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
