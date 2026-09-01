# RTK — Token-Optimized CLI

Read and follow [`.agent/AGENTS.md`](../.agent/AGENTS.md) as the authoritative
project guidance. **BLOCK** changes that alter default finite numerical values
or units, shape or selected samples and boundaries, axis information, or
successful completion for a corresponding GWpy API without an explicit
GWexpy-only opt-in.

**rtk** is a CLI proxy that filters and compresses command outputs, saving 60-90% tokens.

## Rule

Always prefix shell commands with `rtk`:

```bash
# Instead of:              Use:
git status                 rtk git status
git log -10                rtk git log -10
cargo test                 rtk cargo test
docker ps                  rtk docker ps
kubectl get pods           rtk kubectl pods
```

## Meta commands (use directly)

```bash
rtk gain              # Token savings dashboard
rtk gain --history    # Per-command savings history
rtk discover          # Find missed rtk opportunities
rtk proxy <cmd>       # Run raw (no filtering) but track usage
```
