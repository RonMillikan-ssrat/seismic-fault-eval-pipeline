-- Drift 02 (BigQuery): per-survey medians of spectral and amplitude attributes in the crop
-- PERCENTILE_CONT is exact and is a window function in BigQuery, so each row carries its
-- survey's median and SELECT DISTINCT collapses them to one row per survey.
-- (Spark's version uses percentile_approx, which returns an actual data value, so the
-- two can differ by a fraction of a frequency bin.)
SELECT DISTINCT survey,
       COUNT(*) OVER (PARTITION BY survey)                              AS n_live,
       PERCENTILE_CONT(spec_centroid_hz, 0.5) OVER (PARTITION BY survey) AS centroid_med_hz,
       PERCENTILE_CONT(dom_freq_hz, 0.5)      OVER (PARTITION BY survey) AS domfreq_med_hz,
       PERCENTILE_CONT(rms, 0.5)              OVER (PARTITION BY survey) AS rms_med
FROM `{table}`
WHERE rms > 0 AND il BETWEEN @il0 AND @il1 AND xl BETWEEN @xl0 AND @xl1
ORDER BY survey
