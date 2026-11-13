# Gemini view-label audit (Study 2 lite)

30 real composites, 15 that Gemini accepted and 15 where it abstained, interleaved
so you can't tell which is which. All of them have COLMAP camera poses, so your
labels can also be checked against the true camera angles.

1. Open `images/<stem>.jpg` (local only, never committed: MVImgNet terms).
2. Fill `horizontal_view` and `side` for every tile in `label_sheet.csv`:
   * `horizontal_view`: front, front_three_quarter, side, rear_three_quarter, back,
     top, bottom, indeterminate
   * `side`: left, right (for side and three-quarter views), neither (front, back,
     top, bottom), indeterminate
3. "Left" means **the object's own left**, as if you were the object. Gemini's
   prompt never defined this, so the score is also reported ignoring left/right.
4. Use `indeterminate` when the object has no meaningful front (a ball, a plain
   box) or you genuinely can't tell. That is what Gemini was told to do.
5. Label each tile on its own. Don't reason "the next frame must be further round".

Then run `python -m analysis.gemini_audit score`.
