-- 06 Gaps and islands: contiguous runs of crosslines on each inline
--
-- Mechanics
--   Number the traces along each inline with ROW_NUMBER (1, 2, 3, ...).
--   For a contiguous run, xl and step * row_number go up together, so
--   xl - step * row_number stays the same. A gap shifts it.
--   That difference (grp) labels each run, and GROUP BY il, grp gives one row per
--   island with its start, end and length. An inline with no gaps has one island.
--   DISTINCT first, so duplicate traces (query 02) don't break up runs.
WITH t AS (
  SELECT DISTINCT il, xl FROM traces_hdr WHERE survey = :survey
),
r AS (
  SELECT il, xl,
         xl - :xl_step * ROW_NUMBER() OVER (PARTITION BY il ORDER BY xl) AS grp
  FROM t
)
SELECT il, MIN(xl) AS xl_start, MAX(xl) AS xl_end, COUNT(*) AS n_traces
FROM r
GROUP BY il, grp
ORDER BY il, xl_start
