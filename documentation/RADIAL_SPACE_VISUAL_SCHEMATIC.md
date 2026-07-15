# Radial Space Visualization — Build Schematic

**Purpose:** Reproduce EXACTLY the dual-panel radial space demo. Two side-by-side 3D dot-matrix models sharing synchronized time. No logic, no ML — purely visual. An AI agent should be able to read this document and produce a pixel-perfect recreation.

---

## Global layout

Two panels side by side, separated by a 1px vertical divider line using the host border color. Each panel takes 50% width. Both panels share a single `requestAnimationFrame` loop and a shared `time` variable that increments by `0.016` per frame (roughly 60fps). Time never resets — it counts up forever.

**Panel titles** (centered above each canvas):
- Left: `"ground truth moves"`
- Right: `"lenses move"`
- Font: 13px, weight 500, primary text color.

---

## Coordinate system (shared by both panels)

### Camera
- Projection: perspective (not orthographic).
- Camera angle: X rotation = `0.0` (perfectly level), Y rotation = `0.6` radians (fixed initial orbit).
- Distance: `22` units from origin.
- Projection formula: for a 3D point `[x, y, z]` after camera rotation:
  - Apply camera Y rotation: `x' = x*cos(0.6) + z*sin(0.6)`, `z' = -x*sin(0.6) + z*cos(0.6)`, `y' = y`.
  - Screen X = `canvasWidth/2 + x' * scale`, Screen Y = `canvasHeight/2 - y' * scale`.
  - `scale = canvasHeight * 0.6 / (distance + z')`.

### Axes
Three axis lines drawn from origin `[0,0,0]` outward, length `8` units each:
- X axis: `[0,0,0]` → `[8,0,0]`, color `#ef4444` (red), label `"X"`.
- Y axis: `[0,0,0]` → `[0,8,0]`, color `#3b82f6` (blue), label `"Y"`.
- Z axis: `[0,0,0]` → `[0,0,8]`, color `#f59e0b` (amber), label `"Z"`.
- Axis stroke width: 2px.
- Label font: 22px, weight 600, sans-serif. Positioned 6px right and 6px below the axis endpoint.

### Origin marker
- Solid red dot (`#ef4444`), radius 5px, drawn at projected `[0,0,0]`.
- Red halo ring around it: radius 10px, stroke `rgba(239, 68, 68, 0.4)`, stroke width 2px, no fill.

---

## Dot geometry

### Ground truth base grid
- Shape: `10 x 30 x 10` (X × Y × Z) — a tall vertical column.
- Spacing: `1.2` units between dots on all axes.
- Center: origin `[0,0,0]`. Each dot positioned at `(i - N/2 + 0.5) * 1.2` per axis.
- Total points: 3,000.
- Color: green `rgba(34, 197, 94, alpha)`.

### Lens 1 base grid
- Shape: `10 x 10 x 10` cube.
- Spacing: `1.2` units.
- Center: origin.
- Pre-rotated `15°` on Y axis from origin (apply Y rotation matrix with angle `15 * π / 180` to every point).
- No X or Z rotation.
- Color: blue `rgba(59, 130, 246, alpha)`.

### Lens 2 base grid
- Shape: `10 x 10 x 10` cube.
- Spacing: `1.2` units.
- Center: origin.
- Pre-rotated `15°` on X axis FIRST, THEN `45°` on Y axis (order matters).
  - X rotation: `y' = y*cos(15°) - z*sin(15°)`, `z' = y*sin(15°) + z*cos(15°)`, `x' = x`.
  - Then Y rotation: `x'' = x'*cos(45°) + z'*sin(45°)`, `z'' = -x'*sin(45°) + z'*cos(45°)`, `y'' = y'`.
- Color: yellow `rgba(234, 179, 8, alpha)`.

---

## Visible window clipping

Only the center `10 x 1.2 = 12` units of the Y axis are visible. The ground truth column is 30 layers tall (`30 * 1.2 = 36` units), but dots outside the range `[-6, +6]` on Y are culled (not drawn). This creates a seamless flow — new dots enter from below, exit above (or vice versa for the lens-moves panel).

---

## Temporal flow behavior

### Time variable
- `time` increments by `0.016` each frame.
- Y shift = `(time * 1.5) % totalHeight` where `totalHeight = 30 * 1.2 = 36`.

### Panel 1 — "ground truth moves"
- The ground truth dots shift upward on Y by the time-based offset.
- For each ground truth dot: `newY = baseY + yShift`. Wrap with modulo so dots loop: if `newY > 18`, subtract `36`; if `newY < -18`, add `36`. Cull if outside `[-6, +6]`.
- Lens 1 and Lens 2 are STATIONARY — drawn at their pre-rotated positions, never modified.
- If the Y rotation toggle is checked, apply an additional Y rotation (`yRot += 0.008` per frame) to the ground truth dots AFTER the temporal shift.

### Panel 2 — "lenses move"
- The ground truth is STATIONARY — drawn as a plain `10 x 10 x 10` cube at origin, no movement.
- Lens 1 and Lens 2 are built as `10 x 30 x 10` tall columns (same as ground truth's base shape), pre-rotated by their respective angles, then shifted upward by the SAME time-based offset.
- Same wrapping and clipping logic as Panel 1, but applied to the lens dots instead.
- If the Y rotation toggle is checked, apply the additional Y rotation to the LENS dots after temporal shift.

---

## Dot rendering

### Depth sorting
All dots across all three layers (ground truth + lens 1 + lens 2) are collected into a single array, projected to screen coordinates, then sorted by Z depth (back to front). Draw in sorted order so closer dots occlude farther ones.

### Per-dot appearance
- Alpha (opacity): `0.3 + 0.7 * ((z + 10) / 20)`. Clamp to `[0.2, 1.0]`.
  - Dots farther from camera (higher Z after camera rotation) are more transparent.
- Size (radius in canvas pixels): `max(1.5, 4 - z * 0.08)`.
  - Dots farther from camera are smaller.
- Shape: filled circle (`arc`, full `2π`).
- No stroke, no outline.

### Color by type
- Ground truth: `rgba(34, 197, 94, alpha)` — green.
- Lens 1: `rgba(59, 130, 246, alpha)` — blue.
- Lens 2: `rgba(234, 179, 8, alpha)` — yellow.

---

## Controls

### Per-panel legend (top-left of each canvas)
Three rows, each with a colored dot indicator and label:
- `●` green (8px circle) + `"ground truth"`
- `●` blue (8px circle) + `"lens 1"`
- `●` yellow (8px circle) + `"lens 2"`
- Font: 11px, secondary text color. Line height: 1.9.

### Per-panel Y rotation toggle (centered below each canvas)
- Label: `"+ Y rotation"`, font 12px, secondary text color.
- Checkbox: 16px × 16px.
- When checked, adds `0.008` radians per frame to the Y rotation angle for whichever layer is moving in that panel (ground truth in Panel 1, lenses in Panel 2).

### Per-panel time display (bottom-left of each canvas)
- Text: `"t = "` + time value to 2 decimal places.
- Font: 11px, muted text color.

---

## Canvas setup

Each panel's canvas:
- Canvas pixel dimensions: `containerWidth * 2` × `500 * 2` (2x for retina).
- CSS dimensions: `width: 100%`, `height: 500px`.
- Background: transparent (inherits from host).
- No user interaction (no drag, no scroll zoom) — view is locked.

---

## Animation loop

Single `requestAnimationFrame` loop drives both panels:
```
function loop() {
    panel1.tick()    // increment time, increment yRot if toggle checked
    panel2.tick()    // same time variable, independent yRot
    panel1.draw()    // clear canvas, draw axes, project & sort & draw all dots, draw origin
    panel2.draw()    // same
    requestAnimationFrame(loop)
}
```

Both panels share the SAME `time` variable. Each panel has its OWN `yRot` variable controlled by its own checkbox.

---

## Rotation math reference

### Y-axis rotation (used for lens pre-rotation and temporal Y spin)
```
x' = x * cos(angle) + z * sin(angle)
y' = y  (unchanged)
z' = -x * sin(angle) + z * cos(angle)
```

### X-axis rotation (used for Lens 2 pre-rotation)
```
x' = x  (unchanged)
y' = y * cos(angle) - z * sin(angle)
z' = y * sin(angle) + z * cos(angle)
```

### Application order for Lens 2
1. Apply X rotation (15°) first.
2. Apply Y rotation (45°) to the result.

---

## Color reference table

| Element | Hex | RGBA template |
|---|---|---|
| Ground truth dots | `#22c55e` | `rgba(34, 197, 94, alpha)` |
| Lens 1 dots | `#3b82f6` | `rgba(59, 130, 246, alpha)` |
| Lens 2 dots | `#eab308` | `rgba(234, 179, 8, alpha)` |
| X axis | `#ef4444` | — |
| Y axis | `#3b82f6` | — |
| Z axis | `#f59e0b` | — |
| Origin dot | `#ef4444` | — |
| Origin halo | `rgba(239, 68, 68, 0.4)` | — |

---

## What this visualization demonstrates

- **Left panel:** Data (ground truth) flows through stationary lenses. This is how GENREG processes static data — the input moves through fixed activation function perspectives. Each lens sees different slices of the data as it passes through.

- **Right panel:** Lenses move through stationary data. This is the inverted view — the perspectives rotate around fixed data. Equivalent mathematically, but visually demonstrates that it is the RELATIVE motion between data and lens that creates signal diversity.

- **Y rotation toggle:** Adds spin on top of temporal flow, demonstrating how rotation and temporal shift compound to create views that never repeat.

- **The origin (red dot):** The shared anchor point. All three grids share this origin. The lenses are rotated FROM the origin, not translated away from it.
