-- 04 Neighbor RMS jumps along each inline (BigQuery)
-- Same logic as the Spark version. Uses a named WINDOW clause so the two LAG calls
-- share one definition (Spark SQL supports WINDOW too).
WITH live AS (
  SELECT il, xl, rms FROM `{ds}.trace_stats` WHERE survey = @survey AND rms > 0
),
nb AS (
  SELECT il, xl, rms,
         LAG(xl)  OVER w AS prev_xl,
         LAG(rms) OVER w AS prev_rms
  FROM live
  WINDOW w AS (PARTITION BY il ORDER BY xl)
)
SELECT il, prev_xl, xl, prev_rms, rms, rms / prev_rms AS ratio
FROM nb
WHERE prev_rms IS NOT NULL
  AND (rms / prev_rms > @jump OR rms / prev_rms < 1.0 / @jump)
ORDER BY il, xl
