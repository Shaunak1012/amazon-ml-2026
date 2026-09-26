"""Stdlib-only upload of one file to shreyas-gpu/out/<key> through the presigned POST policy (usable before any venv).

    python3 scripts/gpu/post_file.py post.json bootstrap.log boot.log
"""
import json
import sys
import urllib.request
import uuid

spec = json.load(open(sys.argv[1]))
key, path = "shreyas-gpu/out/" + sys.argv[2], sys.argv[3]
data = open(path, "rb").read()
b = uuid.uuid4().hex
parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
         for k, v in {**spec["fields"], "key": key}.items()]
parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="f"\r\n'
             f'Content-Type: application/octet-stream\r\n\r\n'.encode() + data + b"\r\n")
parts.append(f"--{b}--\r\n".encode())
req = urllib.request.Request(spec["url"], data=b"".join(parts),
                             headers={"Content-Type": f"multipart/form-data; boundary={b}"})
urllib.request.urlopen(req, timeout=300)
