# Submission ledger

Daily limit: **TBD** (from COMPETITION.md) · Reset time: **TBD IST** · Reserve ≥ 2 for the final day.

Gate before each submission (see CLAUDE.md → Submission discipline):
1. CV improved meaningfully over best submitted run (> fold std)? 2. Remaining today > reserve?
3. Code + config committed? 4. Validator passes? → `python scripts/tag_submission.py <file> --exp <id> --cv <cv> --sample <sample>`

| # | timestamp (IST) | experiment id | git tag | CV | public LB | remaining today | file |
|---|---|---|---|---|---|---|---|
