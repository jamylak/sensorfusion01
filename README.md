# Blueprint Canvas

Technical schematic explorer built in C with Raylib.

## Structure

- `include/blueprint.h`: engine types, camera helpers, registry API, drawing primitives.
- `src/blueprint.c`: camera update, layered renderer, procedural world grid, primitive drawing.
- `src/main.c`: demo content and application bootstrap.
- `Makefile`: build and run targets.

## Controls

- `q`: quit
- `space`: pause or resume signal animation
- `r`: reset camera and UI state
- `1`, `2`: switch pages
- `mouse drag`: pan
- `mouse wheel`: zoom around cursor
- `h`, `j`, `k`, `l`: fine pan nudges

## Build

```sh
make run
```

If Raylib is not in the default Homebrew path, set `RAYLIB_DIR=/path/to/raylib`.
