# Public jump-training artifacts

The checkpoints, exported ONNX policies, TensorBoard event files, evaluation
arrays, supervisor state, and run metadata used by the jump experiments are
published with the GitHub release
[`jump-training-v1-v8-2026-09-24`](https://github.com/FZAbiding/microduck_rl/releases/tag/jump-training-v1-v8-2026-09-24).

They are release assets instead of Git objects because the complete dataset is
about 5.1 GiB. Keeping it out of Git history makes normal clones small while
leaving the training record publicly downloadable.

The release contains one archive for each top-level `artifacts/` experiment
group, one archive for `logs/`, and `SHA256SUMS`. From the repository root,
restore any downloaded archive with:

```bash
tar -xzf artifacts-jump-v8.tar.gz
tar -xzf logs.tar.gz
sha256sum -c SHA256SUMS
```

The archives retain their repository-relative paths, so extraction recreates
`artifacts/...` and `logs/...` directly. The independent local checkout under
`third_party/` and all virtual environments are intentionally excluded.

The code associated with the release is commit `20bbe80` on `develop`. Its CPU
test result before publication was `281 passed, 1 skipped`.
