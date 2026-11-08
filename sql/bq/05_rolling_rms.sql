-- 05 RMS against the local average along the inline (BigQuery)
-- Same logic as the Spark version. BigQuery supports both ROWS and RANGE frames;
-- RANGE needs a single numeric ORDER BY key, which xl (INT64) is.
WITH live AS (
  SELECT il, xl, rms FROM `{ds}.trace_stats` WHERE survey = @survey AND rms > 0
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
WHERE rms / roll_range_excl_self > @spike OR rms / roll_range_excl_self < 1.0 / @spike
ORDER BY il, xl
