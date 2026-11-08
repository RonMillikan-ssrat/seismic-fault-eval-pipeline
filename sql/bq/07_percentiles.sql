-- 07 Attribute percentiles per survey (BigQuery)
-- Difference from the Spark version:
--   percentile_approx(x, array(0.01, 0.5, 0.99), accuracy)
--     -> APPROX_QUANTILES(x, 100), which returns 101 boundaries (min, 1%, ..., 99%, max);
--        pick entries with [OFFSET(k)]. The precision is fixed by BigQuery rather
--        than set by an accuracy argument.
--   [a, b, c] builds an ARRAY literal.
WITH q AS (
  SELECT survey,
         COUNT(*) AS n_live,
         APPROX_QUANTILES(rms, 100)              AS rq,
         APPROX_QUANTILES(max_abs, 100)          AS mq,
         APPROX_QUANTILES(dom_freq_hz, 100)      AS dq,
         APPROX_QUANTILES(spec_centroid_hz, 100) AS cq
  FROM `{ds}.trace_stats`
  WHERE rms > 0
  GROUP BY survey
)
SELECT survey, n_live,
       [rq[OFFSET(1)], rq[OFFSET(50)], rq[OFFSET(99)]] AS rms_p01_p50_p99,
       [mq[OFFSET(1)], mq[OFFSET(50)], mq[OFFSET(99)]] AS maxabs_p01_p50_p99,
       [dq[OFFSET(1)], dq[OFFSET(50)], dq[OFFSET(99)]] AS domfreq_p01_p50_p99,
       [cq[OFFSET(1)], cq[OFFSET(50)], cq[OFFSET(99)]] AS centroid_p01_p50_p99
FROM q
ORDER BY survey
