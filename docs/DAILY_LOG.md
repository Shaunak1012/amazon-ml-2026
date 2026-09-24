# Daily log

Short entries at the end of each block, newest first: **tried / worked / next**. A teammate joining
mid-way should catch up in 2 minutes. Also keep the GPU queue current.

## GPU queue (update whenever it changes)
| now running | ETA (IST) | next up | owner |
|---|---|---|---|
| — | — | — | — |

## 2026-09-24 (pre-launch)
- **Did:** built scaffold — metrics/CV/OOF/submission validator with tests, heartbeat + watchdog + Discord alerts, playbooks.
- **Worked:** tests green; monitor demo fires alerts for clean run, OOM crash, NaN, hang, hard kill; live Discord
  alerts verified. Image downloader tested locally. 6 backbones (3.6 GB) pre-cached and verified to load offline on
  GPU in bf16: e5-base-v2, bge-base-en-v1.5, MiniLM-L6, deberta-v3-base, siglip-base-224, clip-vit-large-14.
- **Next (Day 1, 00:00 IST):** follow the Day-1 checklist in CLAUDE.md.
