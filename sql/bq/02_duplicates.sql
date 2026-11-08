-- 02 Duplicate traces (BigQuery)
-- Difference from the Spark version: QUALIFY filters on a window function in the
-- same query, so no outer query is needed. QUALIFY is evaluated after WHERE,
-- GROUP BY, HAVING and the window functions, and it can use the select alias rn.
SELECT il, xl, trace_seq,
       ROW_NUMBER() OVER (PARTITION BY il, xl ORDER BY trace_seq) AS rn,
       COUNT(*)     OVER (PARTITION BY il, xl)                    AS copies
FROM `{ds}.traces`
WHERE survey = @survey
QUALIFY rn > 1
ORDER BY il, xl, rn
