-- 02 Duplicate traces: more than one trace at the same (il, xl)
--
-- Mechanics
--   ROW_NUMBER() numbers the rows inside each (il, xl) group, oldest trace first.
--   rn = 1 is the copy to keep; rn > 1 are the extra copies.
--   COUNT(*) OVER the same window gives the group size on every row, without
--   collapsing rows the way GROUP BY would.
--   A WHERE clause can't reference a window function directly, so the filter goes
--   in an outer query. BigQuery (and Spark 4.x) can do it in one step with
--   QUALIFY rn > 1.
WITH ranked AS (
  SELECT trace_seq, il, xl,
         ROW_NUMBER() OVER (PARTITION BY il, xl ORDER BY trace_seq) AS rn,
         COUNT(*)     OVER (PARTITION BY il, xl)                    AS copies
  FROM traces_hdr
  WHERE survey = :survey
)
SELECT il, xl, trace_seq, rn, copies
FROM ranked
WHERE rn > 1
ORDER BY il, xl, rn
