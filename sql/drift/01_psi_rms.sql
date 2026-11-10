-- Drift 01: Population Stability Index (PSI) of trace RMS, each survey vs a reference
--
-- PSI = sum over bins of (p_survey - p_ref) * ln(p_survey / p_ref)
--   Bins are the reference survey's deciles, so the reference has ~10% in each bin.
--   Common rules of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift.
--
-- Two versions:
--   raw   PSI of RMS as recorded. A constant gain moves every trace into the top
--         bins, so PSI is large even though the model (which normalizes its input)
--         sees no change.
--   norm  PSI of RMS divided by the survey's median RMS: normalizes the monitored
--         quantity the same way the model normalizes its input, removing constant gain.
--
-- Mechanics
--   percentile_approx(x, array(...)) returns all nine decile edges in one pass.
--   size(filter(edges, e -> rms > e)) counts how many edges a value exceeds, which is
--   its bin number 0..9 (a higher-order function: no join to a bins table).
--   grid + LEFT JOIN makes empty bins explicit; GREATEST(p, 1e-4) keeps ln() finite.
--   Only traces inside the crop (:il0..:il1, :xl0..:xl1) are compared, so a survey
--   covering more area than the reference does not show up as drift.
WITH stats AS (
  SELECT survey, rms FROM trace_stats
  WHERE rms > 0 AND il BETWEEN :il0 AND :il1 AND xl BETWEEN :xl0 AND :xl1
),
med AS (
  SELECT survey, percentile_approx(rms, 0.5, 10000) AS med FROM stats GROUP BY survey
),
normed AS (
  SELECT s.survey, s.rms, s.rms / m.med AS rms_n FROM stats s JOIN med m ON s.survey = m.survey
),
edges AS (
  SELECT percentile_approx(rms,   array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 10000) AS e_raw,
         percentile_approx(rms_n, array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 10000) AS e_norm
  FROM normed WHERE survey = :reference
),
binned AS (
  SELECT n.survey,
         size(filter(e.e_raw,  x -> n.rms   > x)) AS b_raw,
         size(filter(e.e_norm, x -> n.rms_n > x)) AS b_norm
  FROM normed n CROSS JOIN edges e
),
counts AS (
  SELECT survey, 'raw' AS kind, b_raw AS bin, COUNT(*) AS n FROM binned GROUP BY survey, b_raw
  UNION ALL
  SELECT survey, 'norm' AS kind, b_norm AS bin, COUNT(*) AS n FROM binned GROUP BY survey, b_norm
),
props AS (
  SELECT survey, kind, bin, n / SUM(n) OVER (PARTITION BY survey, kind) AS p FROM counts
),
grid AS (
  SELECT s.survey, k.kind, b.bin
  FROM (SELECT DISTINCT survey FROM counts) s
  CROSS JOIN (SELECT explode(array('raw', 'norm')) AS kind) k
  CROSS JOIN (SELECT explode(sequence(0, 9)) AS bin) b
),
filled AS (
  SELECT g.survey, g.kind, g.bin, GREATEST(COALESCE(p.p, 0.0), 1e-4) AS p
  FROM grid g LEFT JOIN props p ON g.survey = p.survey AND g.kind = p.kind AND g.bin = p.bin
)
SELECT a.survey,
       SUM(CASE WHEN a.kind = 'raw'  THEN (a.p - r.p) * ln(a.p / r.p) END) AS psi_rms_raw,
       SUM(CASE WHEN a.kind = 'norm' THEN (a.p - r.p) * ln(a.p / r.p) END) AS psi_rms_norm
FROM filled a
JOIN filled r ON r.survey = :reference AND a.kind = r.kind AND a.bin = r.bin
GROUP BY a.survey
ORDER BY a.survey
