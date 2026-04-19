# Vendored intersphinx inventories

This directory stores local copies of dependency `objects.inv` files for Sphinx.

Why these files are committed:

- local development in this repository often runs without outbound network access
- CI has previously failed when upstream inventory URLs returned errors
- Sphinx still links users to the official external documentation URLs; only the symbol inventory is local

How to refresh:

```bash
python scripts/update_intersphinx_inventories.py --write
```

How to check upstream reachability without changing files:

```bash
python scripts/update_intersphinx_inventories.py --check-upstream
```

Inventory source metadata is recorded in `sources.json`.
