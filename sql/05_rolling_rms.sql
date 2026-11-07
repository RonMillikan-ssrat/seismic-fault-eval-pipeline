-- 05 Traces whose RMS departs from the local average along the inline
--
-- Mechanics: ROWS vs RANGE frames
--   ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING   = the 10 neighboring *rows* on
--       each side, however far apart their crossline numbers are. Next to a gap,
--       the window reaches farther along the line.
--   RANGE BETWEEN 10 PRECEDING AND 10 FOLLOWING  = rows whose *xl value* is within
--       +-10 of this row's xl. Gaps just mean fewer traces in the window, so the
--       window always covers the same distance on the ground.
--   Both are computed so you can compare. The flag uses RANGE, and excludes the
--   trace itself from its own average (sum minus self, count minus self).
--   BigQuery supports both frames (RANGE needs a single numeric ORDER BY key).
WITH live AS (
  SELECT il, xl, rms FROM trace_stats WHERE survey = :survey AND rms > 0
),
w AS (
  SELECT il, xl, rms,
         AVG(rms) OVER (PARTITION BY il ORDER BY xl
                        ROWS  BETWEEN 10 PRECEDING AND 10 FOLLOWING) AS roll_rows,
         SUM(rms) OVER (PARTITION BY il ORDER BY xl
                        RANGE BETWEEN 10 PRECEDING AND 10 FOLLOWING) AS sum_range,
         COUNT(*) OVER (PARTITION BY il ORDER BY xl
                        RANGE BETWEEN 10 PRECEDING AND 10 FOLLOWING) AS n_range
  FROM live
),
f AS (
  SELECT il, xl, rms, roll_rows,
         (sum_range - rms) / NULLIF(n_range - 1, 0) AS roll_range_excl_self
  FROM w
)
SELECT il, xl, rms, roll_rows, roll_range_excl_self,
       rms / roll_range_excl_self AS ratio
FROM f
WHERE rms / roll_range_excl_self > :spike OR rms / roll_range_excl_self < 1.0 / :spike
ORDER BY il, xl
