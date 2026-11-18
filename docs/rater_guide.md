# Rater guide: four-view grids

Thanks for helping. You'll look at about 84 pairs of images. Each image is a 2x2
grid that is supposed to show **one object from four different viewpoints**. Each
pair was generated from the same text prompt. One image comes from one model and
one from another, in random order. You don't know which is which, and neither do
the other raters. Please don't compare notes until everyone is done.

Expect 45-60 minutes. Breaks are fine.

## What you get

```
images/pair-001_A.png, pair-001_B.png, ...
pairwise.csv     one row per pair
scorecard.csv    one row per image
```

Open the two images of a pair side by side. The prompt is in both CSVs.

## pairwise.csv: one question per pair

**`same_object_choice`**: in which grid do the four tiles look most clearly like
*the same physical object*? Enter `A`, `B` or `tie`.

* Judge the object, not the background, lighting or image quality.
* A grid that shows four copies of the *same view* still counts as the same
  object. Duplication is scored separately below, so don't punish it here.
* Use `tie` only when you honestly can't pick. A slight preference is still a
  preference.

`notes` is optional, for anything odd (a tile with no object, text artefacts,
black bars).

## scorecard.csv: a few ratings per image

Tiles are numbered like this:

```
+---+---+
| 1 | 2 |
+---+---+
| 3 | 4 |
+---+---+
```

| Column | What to enter |
| --- | --- |
| `identity_consistency_1_5` | 1 = four different objects, 3 = same kind of object but details change (colour, logo, parts), 5 = clearly the same individual object in every tile |
| `distinct_views_1_4` | how many genuinely different viewpoints you see. Four near-identical front views = 1. Front, front again, side, back = 3. |
| `duplicate_tile_pairs` | which tiles show (nearly) the same viewpoint, as digit pairs separated by spaces, e.g. `12` or `12 34`. Leave empty if all four differ. Small shifts in zoom or position still count as the same view. |
| `prompt_fidelity_1_5` | how well the object matches the prompt's description (colours, parts, text), ignoring the viewpoints |
| `valid_grid` | `1` if all four tiles contain the object, `0` if any tile is empty, cut off, or not part of a 2x2 layout |

## Three quick examples

1. **A bottle shown four times from the front, label identical.** Identity 5,
   distinct views 1, duplicate pairs `12 13 14 23 24 34`.
2. **A backpack: front, left side, back, right side, but the logo changes colour
   in tile 3.** Identity 3 or 4, distinct views 4, duplicate pairs empty.
3. **Four different mugs in four poses.** Identity 1, and the view count is
   whatever you see. It matters less here, but fill it in.

When you're done, send back both CSV files. Rename them if you like, but keep the
columns as they are.
