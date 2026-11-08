-- 03 Dead traces by inline (BigQuery)
-- Differences from the Spark version:
--   SUM(CASE WHEN ...)   ->  COUNTIF(...)
--   collect_list(...)    ->  ARRAY_AGG(... IGNORE NULLS ORDER BY xl): BigQuery arrays
--                            can't contain NULLs, and the order can be set explicitly
SELECT il,
       COUNT(*) AS n_traces,
       COUNTIF(rms = 0 OR zero_frac > 0.99) AS n_dead,
       ARRAY_AGG(IF(rms = 0 OR zero_frac > 0.99, xl, NULL) IGNORE NULLS ORDER BY xl) AS dead_xl
FROM `{ds}.trace_stats`
WHERE survey = @survey
GROUP BY il
HAVING COUNTIF(rms = 0 OR zero_frac > 0.99) > 0
ORDER BY il
