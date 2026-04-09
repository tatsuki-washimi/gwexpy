# GWexpy Failure Analysis Report

## Summary of Failed Runs

| Workflow | Run ID | Timestamp | Status | Short Diagnosis | Priority |
| --- | --- | --- | --- | --- | --- |
| Documentation | [24181669178](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24181669178) | 2026-04-09T08:57:44Z | failure | Sphinx: Intersphinx 404 (GWpy docs stable) | P2 |
| Documentation | [24179768936](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24179768936) | 2026-04-09T08:11:14Z | failure | Sphinx: Intersphinx 404 (GWpy docs stable) | P2 |
| Documentation | [24175997253](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175997253) | 2026-04-09T06:29:24Z | failure | Sphinx: Intersphinx 404 (GWpy docs stable) | P2 |
| Documentation | [24175951705](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175951705) | 2026-04-09T06:28:01Z | failure | Sphinx: Intersphinx 404 (GWpy docs stable) | P2 |
| Documentation | [24175690532](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175690532) | 2026-04-09T06:19:48Z | failure | Sphinx: Intersphinx 404 (GWpy docs stable) | P2 |
| Documentation | [24175538850](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175538850) | 2026-04-09T06:14:52Z | failure | Sphinx: Intersphinx 404 (GWpy docs stable) | P2 |
| Documentation | [24175240855](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175240855) | 2026-04-09T06:05:38Z | failure | Sphinx: Intersphinx 404 (GWpy docs stable) | P2 |
| Documentation | [24175182615](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175182615) | 2026-04-09T06:03:53Z | failure | General: Check log for details | P1 |
| Documentation | [24174874329](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24174874329) | 2026-04-09T05:53:58Z | failure | General: Check log for details | P1 |
| Documentation | [24174235047](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24174235047) | 2026-04-09T05:33:08Z | failure | General: Check log for details | P1 |
| Tests | [24181736907](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24181736907) | 2026-04-09T08:59:22Z | failure | Python: SyntaxError (__future__ position) | P0 |
| Tests | [24181669187](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24181669187) | 2026-04-09T08:57:44Z | failure | Python: SyntaxError (__future__ position) | P0 |
| Tests | [24179768927](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24179768927) | 2026-04-09T08:11:14Z | failure | Python: SyntaxError (__future__ position) | P0 |
| Tests | [24175997222](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175997222) | 2026-04-09T06:29:24Z | failure | Python: SyntaxError (__future__ position) | P0 |
| Tests | [24175951729](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175951729) | 2026-04-09T06:28:01Z | failure | Python: SyntaxError (__future__ position) | P0 |
| Tests | [24175690510](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175690510) | 2026-04-09T06:19:48Z | failure | pytest: missing 'iminuit' dependency | P0 |
| Tests | [24175538851](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175538851) | 2026-04-09T06:14:52Z | failure | pytest: missing 'iminuit' dependency | P0 |
| Tests | [24175240870](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175240870) | 2026-04-09T06:05:38Z | failure | General: Check log for details | P1 |
| Tests | [24175182662](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175182662) | 2026-04-09T06:03:53Z | failure | General: Check log for details | P1 |
| Tests | [24174874354](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24174874354) | 2026-04-09T05:53:58Z | failure | General: Check log for details | P1 |

## Distinct Failure Breakdown

### Python: SyntaxError (__future__ position)

**Priority**: P0

**Affected runs**: [24181736907](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24181736907), [24181669187](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24181669187), [24179768927](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24179768927), [24175997222](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175997222), [24175951729](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175951729)

#### Log Excerpt
```text
test-conda-full	Run unit tests (forked, without GUI)	2026-04-09T09:00:36.4617284Z !!!!!!!!!!!!!!!!!!! Interrupted: 47 errors during collection !!!!!!!!!!!!!!!!!!!
test-conda-full	Run unit tests (forked, without GUI)	2026-04-09T09:00:36.4618014Z 30 skipped, 47 errors in 15.23s
test-conda-full	Run unit tests (forked, without GUI)	2026-04-09T09:00:37.4101634Z ##[error]Process completed with exit code 2.
```

- **Suggested Fix**: Move '__future__' imports to the top of the file, before any other code/imports.
- **Files Affected**: tests/fitting/test_fitting_core.py, tests/fitting/test_gls.py
- **PR Required**: yes

### pytest: missing 'iminuit' dependency

**Priority**: P0

**Affected runs**: [24175690510](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175690510), [24175538851](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175538851)

#### Log Excerpt
```text
test-conda-full	Run unit tests (forked, without GUI)	2026-04-09T06:21:12.4526520Z !!!!!!!!!!!!!!!!!!! Interrupted: 61 errors during collection !!!!!!!!!!!!!!!!!!!
test-conda-full	Run unit tests (forked, without GUI)	2026-04-09T06:21:12.4526867Z 29 skipped, 61 errors in 14.47s
test-conda-full	Run unit tests (forked, without GUI)	2026-04-09T06:21:13.2133831Z ##[error]Process completed with exit code 2.
```

- **Suggested Fix**: Add 'iminuit' to the CI test environment (conda or pip).
- **Files Affected**: pyproject.toml, .github/workflows/test.yml
- **PR Required**: yes

### General: Check log for details

**Priority**: P1

**Affected runs**: [24175182615](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175182615), [24174874329](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24174874329), [24174235047](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24174235047), [24175240870](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175240870), [24175182662](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175182662), [24174874354](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24174874354)

#### Log Excerpt
```text
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.0285345Z ERROR: Could not find a version that satisfies the requirement types-numpy (from versions: none)
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.0798832Z ERROR: No matching distribution found for types-numpy
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.1351398Z ##[error]Process completed with exit code 1.
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.1469849Z Post job cleanup.
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2450128Z [command]/usr/bin/git version
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2490678Z git version 2.53.0
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2543631Z Temporarily overriding HOME='/home/runner/work/_temp/1fa8dc1c-badb-4436-b19b-0c390228a7a4' before making global git config changes
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2545177Z Adding repository directory to the temporary git global config as a safe directory
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2558743Z [command]/usr/bin/git config --global --add safe.directory /home/runner/work/gwexpy/gwexpy
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2596443Z [command]/usr/bin/git config --local --name-only --get-regexp core\.sshCommand
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2629539Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'core\.sshCommand' && git config --local --unset-all 'core.sshCommand' || :"
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2860254Z [command]/usr/bin/git config --local --name-only --get-regexp http\.https\:\/\/github\.com\/\.extraheader
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2890606Z http.https://github.com/.extraheader
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2894527Z [command]/usr/bin/git config --local --unset-all http.https://github.com/.extraheader
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.2925081Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'http\.https\:\/\/github\.com\/\.extraheader' && git config --local --unset-all 'http.https://github.com/.extraheader' || :"
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.3152562Z [command]/usr/bin/git config --local --name-only --get-regexp ^includeIf\.gitdir:
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.3184703Z [command]/usr/bin/git submodule foreach --recursive git config --local --show-origin --name-only --get-regexp remote.origin.url
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.3540482Z Cleaning up orphan processes
build-docs	UNKNOWN STEP	2026-04-09T06:05:52.3826242Z ##[warning]Node.js 20 actions are deprecated. The following actions are running on Node.js 20 and may not work as expected: actions/checkout@v4, conda-incubator/setup-miniconda@v3. Actions will be forced to run with Node.js 24 by default starting June 2nd, 2026. Node.js 20 will be removed from the runner on September 16th, 2026. Please check if updated versions of these actions are available that support Node.js 24. To opt into Node.js 24 now, set the FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true environment variable on the runner or in your workflow file. Once Node.js 24 becomes the default, you can temporarily opt out by setting ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true. For more information see: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/
```

- **Suggested Fix**: Investigate further.
- **Files Affected**: Unknown
- **PR Required**: yes

### Sphinx: Intersphinx 404 (GWpy docs stable)

**Priority**: P2

**Affected runs**: [24181669178](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24181669178), [24179768936](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24179768936), [24175997253](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175997253), [24175951705](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175951705), [24175690532](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175690532), [24175538850](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175538850), [24175240855](https://github.com/tatsuki-washimi/gwexpy/actions/runs/24175240855)

#### Log Excerpt
```text
build-docs	Build documentation (Strict)	2026-04-09T09:02:21.6907786Z sphinx-sitemap: sitemap.xml was generated for URL https://tatsuki-washimi.github.io/gwexpy/docs/ in /home/runner/work/gwexpy/gwexpy/docs/_build/html/docs/sitemap.xml
build-docs	Build documentation (Strict)	2026-04-09T09:02:21.6909577Z [01mbuild finished with problems, 22 warnings (with warnings treated as errors).[39;49;00m
build-docs	Build documentation (Strict)	2026-04-09T09:02:25.6919673Z ##[error]Process completed with exit code 1.
```

- **Suggested Fix**: Update intersphinx_mapping in docs/conf.py to a working URL or handle 404 gracefully.
- **Files Affected**: docs/conf.py
- **PR Required**: maybe

