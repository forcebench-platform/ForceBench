# Release validation

[validation/public_report.json](validation/public_report.json) is the machine-readable record supplied for this export. Its source, environment, asset, data/model identities, commands and stated limitations define the tested scope. The exporter validates coverage and packages that report; it does not execute or independently attest the experiments.

The report records **220 CPU tests passed, 0 failed, 0 errors and 0 skipped**, **10 passed simulator checks covering eight tasks**, and **ten passed model-system checks**. These checks may exercise different software layers. Consult the individual records before treating model-only checks as end-to-end policy rollouts or existing environments as fresh installs.

**Paper-score reproduction is not verified.** Frozen paper tables remain reference values. External assets, datasets and trained checkpoints are not distributed by this source archive. See [validation scope](docs/validation_scope.md) and [NOTICE](NOTICE.md).
