-- 04 Sudden RMS jumps between neighboring traces along each inline
-- Catches gain problems, bad shots/receivers, or a block processed differently.
--
-- Mechanics
--   LAG(rms) OVER (PARTITION BY il ORDER BY xl) returns the previous trace's RMS on
--   the same inline. The first trace on each inline gets NULL, which drops out.
--   Dead traces (rms = 0) are excluded first, so they don't divide by zero or show
--   up as jumps (query 03 reports them).
--   A ratio above :jump (or below 1/:jump) is flagged. Each edge of a scaled block
--   gives one flag per inline: one where the block starts and one where it ends.
--   The same window with PARTITION BY xl ORDER BY il checks along crosslines.
WITH live AS (
  SELECT il, xl, rms FROM trace_stats WHERE survey = :survey AND rms > 0
),
nb AS (
  SELECT il, xl, rms,
         LAG(xl)  OVER (PARTITION BY il ORDER BY xl) AS prev_xl,
         LAG(rms) OVER (PARTITION BY il ORDER BY xl) AS prev_rms
  FROM live
)
SELECT il, prev_xl, xl, prev_rms, rms, rms / prev_rms AS ratio
FROM nb
WHERE prev_rms IS NOT NULL
  AND (rms / prev_rms > :jump OR rms / prev_rms < 1.0 / :jump)
ORDER BY il, xl
