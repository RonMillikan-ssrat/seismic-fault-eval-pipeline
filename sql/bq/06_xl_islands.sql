-- 06 Gaps and islands along each inline (BigQuery)
-- Same logic as the Spark version; only the parameter syntax differs.
WITH t AS (
  SELECT DISTINCT il, xl FROM `{ds}.traces` WHERE survey = @survey
),
r AS (
  SELECT il, xl,
         xl - @xl_step * ROW_NUMBER() OVER (PARTITION BY il ORDER BY xl) AS grp
  FROM t
)
SELECT il, MIN(xl) AS xl_start, MAX(xl) AS xl_end, COUNT(*) AS n_traces
FROM r
GROUP BY il, grp
ORDER BY il, xl_start
