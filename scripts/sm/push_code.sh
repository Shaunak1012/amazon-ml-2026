#!/usr/bin/env bash
# Local: publish the committed shreyas HEAD as the SageMaker box's code snapshot (the box has no GitHub access).
set -euo pipefail
cd "$(dirname "$0")/../.."
B=amazon-sagemaker-548171706001-ap-southeast-2-bjmz86wfr8l6oy
tmp=$(mktemp -d)
git archive --format=tar HEAD -o "$tmp/repo.tar"
git log -1 --format='%h %s' > "$tmp/COMMIT" && tar -rf "$tmp/repo.tar" -C "$tmp" COMMIT
gzip -f "$tmp/repo.tar" && aws s3 cp --only-show-errors "$tmp/repo.tar.gz" "s3://$B/shreyas/code/repo.tar.gz"
aws s3 cp --only-show-errors scripts/sm/bootstrap.sh "s3://$B/shreyas/bootstrap.sh"
echo "published $(cat "$tmp/COMMIT")"; rm -rf "$tmp"
