-- 07 Attribute percentiles for every survey in the lake (no survey filter)
-- Used later as the reference distribution for drift checks.
--
-- Mechanics
--   An exact percentile needs a full sort of the column. percentile_approx keeps a
--   small sketch per partition, merges the sketches, and answers within a known
--   error. The third argument (accuracy) trades memory for precision: relative
--   error is about 1/accuracy.
--   BigQuery equivalent: APPROX_QUANTILES(rms, 100)[OFFSET(50)] for the median.
--   Live traces only (rms > 0), so dead traces don't drag down the low percentiles.
SELECT survey,
       COUNT(*) AS n_live,
       percentile_approx(rms,              array(0.01, 0.5, 0.99), 10000) AS rms_p01_p50_p99,
       percentile_approx(max_abs,          array(0.01, 0.5, 0.99), 10000) AS maxabs_p01_p50_p99,
       percentile_approx(dom_freq_hz,      array(0.01, 0.5, 0.99), 10000) AS domfreq_p01_p50_p99,
       percentile_approx(spec_centroid_hz, array(0.01, 0.5, 0.99), 10000) AS centroid_p01_p50_p99
FROM trace_stats
WHERE rms > 0
GROUP BY survey
ORDER BY survey
