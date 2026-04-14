# Sensor Fusion

This repo is mostly AI-generated and built for fast visualization, not for code quality.


https://github.com/user-attachments/assets/e6bef4df-8447-460a-a1ea-753e23ac8b10


## Controls

- `q`: quit
- `1`-`6`: switch pages between `1` math overview, `2` fusion overview, `3` innovation lab, `4` `H` lab, `5` `R` lab, and `6` `H(x)` lab
- `space`: pause or resume animation on pages `1`-`5`
- `r`: reset the current demo state on the current page
- `mouse wheel`: zoom around cursor
- `left drag`: pan, except while dragging an interactive widget
- `middle drag`: pan anywhere in the viewport
- `h`, `j`, `k`, `l`: fine pan nudges

## Scene-specific controls

- Fusion overview (`2`): `p` toggles the execution timeline, `n` steps forward, `b` steps backward, and `u`/`o` move through saved frames
- Innovation lab (`3`): drag the measurement handle with the left mouse button to override the generated sample; `g` restores the auto-generated measurement
- `H` lab (`4`): hover a matrix cell and use the mouse wheel to adjust it, left click to cycle common values, right click to zero it; `p`, `v`, and `m` load the position, velocity, and mixed presets
- `R` lab (`5`): drag the slider handles to change the covariance entries
- `H(x)` lab (`6`): same matrix editing controls as the `H` lab; `p`, `v`, and `m` load presets, `space` advances one debugger step, and `shift+space` toggles autoplay

## Click-through shortcuts

- In the fusion overview, click a measurement matrix panel to open the matching `H` or `R` lab
- Click an innovation vector panel to open the innovation lab
- Click the predicted state marker to jump straight to the `H(x)` lab

## Dependencies

- `raylib`
- a C compiler such as `cc`/`clang`
- optional: `pkg-config` so the `Makefile` can auto-detect Raylib flags; otherwise it falls back to `RAYLIB_DIR`

## Build

```sh
make
make run
```
