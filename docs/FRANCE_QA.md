# France QA (unseen country) - 27 Sep 2026

France has no labels anywhere, so its quality is checked three ways: public-LB probes on fixed probabilities, a
label-free drift guard between model versions, and a manual audit of sampled decisions.

## 1. Public-LB threshold probes (same probabilities, only the France threshold changes)
| France threshold | 0.55 | 0.75 | 0.90 |
|---|---|---|---|
| Public LB | 0.985959 | 0.98631 | 0.986359 |

Loosening hurts, tightening is flat: the final uses 0.85 (plateau centre).

## 2. Label-free signals by model (test)
| Model | French uncertain pairs / S1 (prob in [0.3, 0.99)) | French matches / S1 |
|---|---|---|
| E023b-full | 0.447 | 3.26 |
| E023b-ST (self-training, e5-base) | 0.365 | 3.29 |
| E029-final (francized e5-large) | 0.379 | 3.27 |
| E030-final (self-trained francized e5-large) | **0.331** | 3.26 |

Drift guard for a model update (`scripts/france_drift.py`): the update must keep the same candidate rows, change the
decided match set of at most 3% of French S1s, and move French matches per S1 by at most 2%.

## 3. Manual audit of E033 French decisions (random samples)
- **Accepted, prob >= 0.95 (833k pairs, the bulk):** all sampled pairs correct. The model handles the French noise
  in this data: "84 RUE de Trevise" vs "84 R. DE TREVISE", region vs department ("Hauts-de-France" vs "Nord"),
  typos, legal form moved or changed (SAS, SARL, EI), upper case, "N°" prefixes.
- **Accepted, 0.85-0.95 (12.5k pairs):** mostly plausible, including records whose name was replaced by an unrelated
  string at the same address; a few dubious (different house number).
- **Rejected, 0.60-0.85 (19.6k pairs):** several look like true matches (name + suffix at the same address); others
  are empty-address names that may belong to another same-name S1.

The remaining French error mass is a small borderline band (~0.1 pairs per French S1) where the text is genuinely
ambiguous. In production this band is the one to route to human review for a new market.
