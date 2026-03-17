# Contributing to U-RNN

Thank you for your interest in contributing! We welcome bug fixes, new dataset support, documentation improvements, and general enhancements.

---

## Ways to Contribute

| Type | How |
|---|---|
| **Bug report** | [Open a Bug Report issue](.github/ISSUE_TEMPLATE/bug_report.yml) |
| **Feature request** | [Open a Feature Request issue](.github/ISSUE_TEMPLATE/feature_request.yml) |
| **Code contribution** | Fork → branch → PR (see below) |
| **New dataset** | Add YAML config + event lists + conversion script in `tools/` |
| **Documentation** | Edit `README.md` or files under `docs/` |

---

## Development Setup

```bash
git clone https://github.com/holmescao/U-RNN
cd U-RNN/U-RNN
conda create -n urnn-dev python=3.8
conda activate urnn-dev
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code runs without error (`python -c "import main; import test; print('OK')"`)
- [ ] Config changes: new parameters added to `config.py` with `--help` description
- [ ] New tools have a module docstring (usage, arguments, output, example)
- [ ] README updated if user-visible behaviour changes
- [ ] No hardcoded hyperparameters (use `args.*` or YAML, not magic numbers)

---

## Code Style

- Python 3.8+, follow PEP 8 loosely
- Function docstrings for public functions
- All hyperparameters via `config.py` argparse — no magic numbers in source

---

## Adding a New Dataset

1. Write a conversion script in `tools/convert_<dataset>_data.py`
2. Add event list files to `src/lib/dataset/<dataset>_train.txt` and `<dataset>_test.txt`
3. Create `configs/<dataset>_scratch.yaml` with calibrated normalization constants
4. Add the dataset to the table in `README.md` Section 2
5. Add an entry to `docs/dataset_knowledge.md`

---

## Questions?

Open a [Feature Request / Question issue](.github/ISSUE_TEMPLATE/feature_request.yml) — we respond to all inquiries.
