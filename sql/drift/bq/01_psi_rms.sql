-- Drift 01 (BigQuery): PSI of trace RMS, each survey vs a reference, raw and normalized
-- Same logic as sql/drift/01_psi_rms.sql. Differences from the Spark version:
--   median           PERCENTILE_CONT(rms, 0.5) OVER (PARTITION BY survey): exact, and a
--                    window function in BigQuery, so it rides on each row
--   decile edges     APPROX_QUANTILES(rms, 10) returns 11 boundaries (min, 10%, ..., max);
--                    offsets 1-9 are the nine inner edges
--   bin number       a correlated subquery over the edge array replaces
--                    size(filter(edges, e -> rms > e))
--   explode(...)     CROSS JOIN UNNEST(...)
--   :name            @name; the table name is substituted as {table} by the caller
-- The decile edges are approximate in both engines, with different algorithms, so PSI
-- values differ slightly between Spark and BigQuery. A survey against itself is exactly 0
-- in both, because the same edges bin both sides.
WITH stats AS (
  SELECT survey, rms FROM `{table}`
  WHERE rms > 0 AND il BETWEEN @il0 AND @il1 AND xl BETWEEN @xl0 AND @xl1
),
normed AS (
  SELECT survey, rms, rms / PERCENTILE_CONT(rms, 0.5) OVER (PARTITION BY survey) AS rms_n
  FROM stats
),
edges AS (
  SELECT APPROX_QUANTILES(rms, 10) AS q_raw, APPROX_QUANTILES(rms_n, 10) AS q_norm
  FROM normed WHERE survey = @reference
),
binned AS (
  SELECT n.survey,
         (SELECT COUNT(*) FROM UNNEST(e.q_raw)  AS x WITH OFFSET o WHERE o BETWEEN 1 AND 9 AND n.rms   > x) AS b_raw,
         (SELECT COUNT(*) FROM UNNEST(e.q_norm) AS x WITH OFFSET o WHERE o BETWEEN 1 AND 9 AND n.rms_n > x) AS b_norm
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
  SELECT s.survey, kind, bin
  FROM (SELECT DISTINCT survey FROM counts) s
  CROSS JOIN UNNEST(['raw', 'norm']) AS kind
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, 9)) AS bin
),
filled AS (
  SELECT g.survey, g.kind, g.bin, GREATEST(COALESCE(p.p, 0.0), 1e-4) AS p
  FROM grid g LEFT JOIN props p ON g.survey = p.survey AND g.kind = p.kind AND g.bin = p.bin
)
SELECT a.survey,
       SUM(IF(a.kind = 'raw',  (a.p - r.p) * LN(a.p / r.p), NULL)) AS psi_rms_raw,
       SUM(IF(a.kind = 'norm', (a.p - r.p) * LN(a.p / r.p), NULL)) AS psi_rms_norm
FROM filled a
JOIN filled r ON r.survey = @reference AND a.kind = r.kind AND a.bin = r.bin
GROUP BY a.survey
ORDER BY a.survey
