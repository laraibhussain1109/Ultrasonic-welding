# Production inspection refactor analysis

## Findings from the previous branch

1. **Actual backend.** Every supplied blower profile selects
   `vit_surface_geometry`; `inspector_factory` therefore constructs
   `TiledViTSurfaceInspector`. PatchCore is a non-authoritative engineering
   comparison backend.
2. **Misleading UI.** The generic training worker selected a status string that
   said `PATCHCORE TRAINING` for every non-TAO backend, even though the factory
   had already selected DINOv2.
3. **Use of the 1,325 images.** The DINO path quality-checked the full list and
   extracted tiled features from its training subset; it did not inherit the
   legacy PatchCore 300-image cap. It did, however, split one ordered sequence
   into fitting and calibration views, then randomly reduced the global token
   pool to the configured memory size.
4. **Leakage.** Adjacent frames from the same rotating physical blower could be
   assigned to fitting and calibration. Those highly correlated views made
   calibration look narrower and more independent than it was.
5. **Memory selection.** Candidate patch tokens from every phase and location
   were concatenated, and a seeded random choice selected the final bank. The
   bank did not guarantee coverage of phases, longitudinal sections, or parts.
6. **Why threshold-only tuning is unsafe.** A broad harmless reflection and a
   tiny fin chip can both create a high appearance residual. Raising a single
   anomaly threshold reduces false rejects by discarding precisely the weak-area,
   high-severity evidence needed for a chip or crack.
7. **Reusable components.** Native overlapping tiling, DINO token extraction,
   bounded CUDA memory search, PCA reconstruction, `structural_descriptor`, ECC
   registration, `broken_fin_mask`, `FinGeometryInspector`, locked native ROI
   capture, and sliding tracking are retained and composed as independent
   evidence branches.

## Implemented production policy

The refactored path separates registered golden appearance, phase/position-aware
DINO memory, golden DINO residual, PCA reconstruction, native fine-fin structure,
global fin geometry, and temporal evidence. Learned appearance alone is a
location-aware candidate—even when two learned branches agree. Confirmed native
structural evidence can fail immediately. Registration or quality failure yields
`VIEW INVALID`, never a defect label.

Dataset splitting is deterministic and physical-group-aware. Nested
`parts/<part-or-session>/` groups cannot cross fitting, calibration, and held-out
validation. Legacy flat data remains supported, but its report explicitly warns
that view-disjoint data is not proof of physical independence.
