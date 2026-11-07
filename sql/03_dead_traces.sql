-- 03 Dead traces, summarized by inline
--
-- Mechanics
--   WHERE filters rows before grouping; HAVING filters groups after aggregation.
--   Here HAVING keeps only the inlines that contain at least one dead trace.
--   A trace counts as dead if its RMS is exactly zero or almost every sample is zero.
--   SUM(CASE WHEN ...) counts matching rows inside each group (BigQuery also has COUNTIF).
SELECT il,
       COUNT(*) AS n_traces,
       SUM(CASE WHEN rms = 0 OR zero_frac > 0.99 THEN 1 ELSE 0 END) AS n_dead,
       collect_list(CASE WHEN rms = 0 OR zero_frac > 0.99 THEN xl END) AS dead_xl
FROM trace_stats
WHERE survey = :survey
GROUP BY il
HAVING SUM(CASE WHEN rms = 0 OR zero_frac > 0.99 THEN 1 ELSE 0 END) > 0
ORDER BY il
