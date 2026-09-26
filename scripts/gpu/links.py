"""LOCAL tool (needs the AWS profile): render the remote-GPU bootstrap and publish GPU jobs with presigned links.

    python scripts/gpu/links.py bootstrap                   # -> prints ONE link; user runs: curl -s "<link>" | bash
    python scripts/gpu/links.py job scripts/gpu/jobs/x.sh   # {{GET:<s3 key>}} -> 7-day presigned GET; -> queue
                                                            # {{GETDIR:<s3 prefix>:<local dir>}} -> one dl line per object
Rendered files (they contain live links) go only to S3, never into the repo.
"""
import base64
import json
import re
import sys
from pathlib import Path

import boto3
from botocore.config import Config

B = "amazon-sagemaker-548171706001-ap-southeast-2-bjmz86wfr8l6oy"
E = 7 * 24 * 3600
s3 = boto3.client("s3", region_name="ap-southeast-2", config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


def get(key: str) -> str:
    return s3.generate_presigned_url("get_object", Params={"Bucket": B, "Key": key}, ExpiresIn=E)


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "bootstrap":
        post = s3.generate_presigned_post(B, "shreyas-gpu/out/${filename}", Fields=None,
                                          Conditions=[["starts-with", "$key", "shreyas-gpu/out/"],
                                                      ["content-length-range", 0, 5 * 1024 ** 3]], ExpiresIn=E)
        t = Path("scripts/gpu/bootstrap.sh.tmpl").read_text()
        t = (t.replace("__CODE_URL__", get("shreyas/code/repo.tar.gz"))
              .replace("__QUEUE_URL__", get("shreyas-gpu/queue/current.sh"))
              .replace("__POST_B64__", base64.b64encode(json.dumps(post).encode()).decode()))
        s3.put_object(Bucket=B, Key="shreyas/private/gpu_bootstrap.sh", Body=t.encode())
        print(get("shreyas/private/gpu_bootstrap.sh"))
    elif cmd == "job":
        t = Path(sys.argv[2]).read_text()

        def getdir(m: re.Match) -> str:
            prefix, local = m.group(1).strip(), m.group(2).strip()
            keys = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=prefix)
                    for o in page.get("Contents", []) if not o["Key"].endswith((".done", "STAGED"))]
            if not keys:
                raise SystemExit(f"nothing under {prefix}")
            return chr(10).join(f'dl "{get(k)}" {local}/{k.rsplit("/", 1)[1]}' for k in keys)

        t = re.sub(r"\{\{GETDIR:([^:}]+):([^}]+)\}\}", getdir, t)
        t = re.sub(r"\{\{GET:([^}]+)\}\}", lambda m: get(m.group(1).strip()), t)
        s3.put_object(Bucket=B, Key="shreyas-gpu/queue/current.sh", Body=t.encode())
        print(f"queued {sys.argv[2]} ({len(t)} bytes)")


if __name__ == "__main__":
    main()
