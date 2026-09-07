# Contributing

This repository is a **completed, unofficial reference implementation** of
Yang et al.'s hybrid CVAE-CGAN protocol. It is not a living product and is
not actively developed.

## What will be considered

- Documentation typos or broken links that make the published protocol harder
  to follow.
- Bugs that prevent a documented command from running, or that make a
  published number impossible to reproduce from the checked-in instructions.

Open an issue first. A pull request without a matching issue may be ignored.

## What will not be accepted

- New generative methods, new backbones, or new loss terms.
- New datasets, private data, or adapters that change the public protocol.
- Feature work, API redesigns, extra metrics, or training-framework ports.
- Changes that would make this repo look affiliated with the original authors.

Fork the repository if you need any of the above. The maintainer may merge
nothing, and issues may go unanswered. That is the expected state, not a
lapse.

## Language

Published non-README files stay in English. Keep the bilingual READMEs in
sync; the English `README.md` is normative.

## Verification

After any code change, run:

```bash
python src/smoke_test.py
```
