-- Drift 02: per-survey medians of spectral and amplitude attributes inside the crop
-- The analysis subtracts the reference survey's row to get shifts (e.g. centroid +6 Hz).
-- Medians rather than means: a few dead or spiky traces don't move them.
SELECT survey,
       COUNT(*)                                        AS n_live,
       percentile_approx(spec_centroid_hz, 0.5, 10000) AS centroid_med_hz,
       percentile_approx(dom_freq_hz,      0.5, 10000) AS domfreq_med_hz,
       percentile_approx(rms,              0.5, 10000) AS rms_med
FROM trace_stats
WHERE rms > 0 AND il BETWEEN :il0 AND :il1 AND xl BETWEEN :xl0 AND :xl1
GROUP BY survey
ORDER BY survey
