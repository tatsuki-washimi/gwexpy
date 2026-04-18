# Gateway Prototype

This directory contains the isolated Phase 3 prototype for the top-level GWexpy gateway page.

## Files

- `index.html`
  - standalone prototype layout
- `gateway.css`
  - scientific/white visual system for the prototype

## Referenced asset

- `docs/_static/images/phase3/gateway_hero_scientific.png`

Generate it with:

```bash
conda run -n gwexpy python scripts/generate_hero_plot.py \
  --output docs/_static/images/phase3/gateway_hero_scientific.png \
  --width 1200 \
  --height 675 \
  --style scientific
```
