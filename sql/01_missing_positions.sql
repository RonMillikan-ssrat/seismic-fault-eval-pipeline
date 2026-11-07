-- 01 Missing (il, xl) positions
-- Build every position the survey grid should have, then keep the ones with no trace.
--
-- Mechanics
--   steps:   the inline/crossline increment, taken as the smallest positive gap
--            between consecutive distinct values (LAG over the sorted distinct list).
--   grid:    sequence() makes an array from min to max; explode() turns it into rows.
--            CROSS JOIN of the two lists = the full rectangular grid.
--   missing: LEFT ANTI JOIN keeps left rows with NO match on the right.
--            Spark-specific syntax. In BigQuery use NOT EXISTS, or
--            LEFT JOIN ... WHERE t.il IS NULL.
--   survey = :survey filters on the partition column, so Spark reads only that
--   survey's files (PartitionFilters in the plan). traces_hdr has no samples
--   column, so the 2 KB of samples per trace is never read (ReadSchema in the plan).
WITH t AS (
  SELECT il, xl FROM traces_hdr WHERE survey = :survey
),
il_steps AS (
  SELECT il - LAG(il) OVER (ORDER BY il) AS d FROM (SELECT DISTINCT il FROM t)
),
xl_steps AS (
  SELECT xl - LAG(xl) OVER (ORDER BY xl) AS d FROM (SELECT DISTINCT xl FROM t)
),
bounds AS (
  SELECT MIN(il) AS il_min, MAX(il) AS il_max, MIN(xl) AS xl_min, MAX(xl) AS xl_max,
         (SELECT MIN(d) FROM il_steps WHERE d > 0) AS il_step,
         (SELECT MIN(d) FROM xl_steps WHERE d > 0) AS xl_step
  FROM t
),
grid AS (
  SELECT i.il, x.xl
  FROM (SELECT explode(sequence(il_min, il_max, il_step)) AS il FROM bounds) i
  CROSS JOIN (SELECT explode(sequence(xl_min, xl_max, xl_step)) AS xl FROM bounds) x
)
SELECT g.il, g.xl
FROM grid g
LEFT ANTI JOIN t ON g.il = t.il AND g.xl = t.xl
ORDER BY g.il, g.xl
