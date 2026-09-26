"""OW00: can the r7i CPU train/run the listwise owner model in time? multilingual-e5-small train + inference
throughput (fp32 vs bf16 autocast) at owner-model sequence lengths, plus real token lengths of record texts."""
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.er_data import cache_dir

print(open("/proc/cpuinfo").read().count("processor"), "vcpus; amx:", "amx" in open("/proc/cpuinfo").read())
print("torch", torch.__version__, "threads", torch.get_num_threads())
name = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
s = pd.read_parquet(cache_dir() / "train_s2.parquet", columns=["business_name", "business_address"]).sample(5000, random_state=0)
lens = np.array([len(tok(f"{n} | {a}")["input_ids"]) for n, a in zip(s.business_name, s.business_address)])
print("record token length p50/p90/p99/max:", np.percentile(lens, [50, 90, 99]).tolist(), lens.max())
for L in (128, 224):
    B = 32
    ids = torch.randint(5, 20000, (B, L))
    mask = torch.ones_like(ids)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for mode in ("fp32", "bf16"):
        on = mode == "bf16"
        model.train()
        for i in range(5):
            if i == 1:
                t = time.time()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=on):
                out = model(input_ids=ids, attention_mask=mask).last_hidden_state
            out.float().pow(2).mean().backward()
            opt.step()
            opt.zero_grad()
        tr = 4 * B / (time.time() - t)
        model.eval()
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=on):
            model(input_ids=ids, attention_mask=mask)
            t = time.time()
            for _ in range(4):
                model(input_ids=ids, attention_mask=mask)
        inf = 4 * B / (time.time() - t)
        print(f"L={L} {mode}: train {tr:.1f} seq/s, inference {inf:.1f} seq/s", flush=True)
print("BENCH OK")
