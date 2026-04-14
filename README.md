# Sensor Fusion

This repo is mostly AI-generated and built for fast visualization, not for code quality.

## Controls

- `q`: quit
- `space`: pause or resume signal animation
- `r`: reset camera and UI state
- `1`, `2`: switch pages
- `mouse drag`: pan
- `mouse wheel`: zoom around cursor
- `h`, `j`, `k`, `l`: fine pan nudges

## Dependencies

- `raylib`
- a C compiler such as `cc`/`clang`
- optional: `pkg-config` so the `Makefile` can auto-detect Raylib flags; otherwise it falls back to `RAYLIB_DIR`

## Build

```sh
make
make run
```
