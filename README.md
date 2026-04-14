# Sensor Fusion

This repo is mostly AI-generated and built for fast visualization, not for code quality.


https://github.com/user-attachments/assets/e6bef4df-8447-460a-a1ea-753e23ac8b10


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
