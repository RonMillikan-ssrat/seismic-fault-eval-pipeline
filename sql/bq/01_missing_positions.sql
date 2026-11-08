-- 01 Missing (il, xl) positions (BigQuery)
-- Differences from the Spark version (sql/01_missing_positions.sql):
--   sequence() + explode()   ->  GENERATE_ARRAY() + CROSS JOIN UNNEST()
--   LEFT ANTI JOIN           ->  NOT EXISTS (BigQuery has no anti-join keyword)
--   :survey                  ->  @survey
--   Table names can't be query parameters, so the runner substitutes {ds}
--   (project.dataset) after checking it is a valid identifier.
--   CROSS JOIN UNNEST(...) is a correlated join: the array is built from the
--   columns of the row on its left, so each bounds row expands into the grid.
WITH t AS (
  SELECT il, xl FROM `{ds}.traces` WHERE survey = @survey
),
il_steps AS (
  SELECT il - LAG(il) OVER (ORDER BY il) AS d FROM (SELECT DISTINCT il FROM t)
),
xl_steps AS (
  SELECT xl - LAG(xl) OVER (ORDER BY xl) AS d FROM (SELECT DISTINCT xl FROM t)
),
bounds AS (
  SELECT MIN(il) AS il_min, MAX(il) AS il_max, MIN(xl) AS xl_min, MAX(xl) AS xl_max FROM t
),
steps AS (
  SELECT (SELECT MIN(d) FROM il_steps WHERE d > 0) AS il_step,
         (SELECT MIN(d) FROM xl_steps WHERE d > 0) AS xl_step
),
grid AS (
  SELECT il, xl
  FROM bounds
  CROSS JOIN steps
  CROSS JOIN UNNEST(GENERATE_ARRAY(il_min, il_max, il_step)) AS il
  CROSS JOIN UNNEST(GENERATE_ARRAY(xl_min, xl_max, xl_step)) AS xl
)
SELECT g.il, g.xl
FROM grid g
WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.il = g.il AND t.xl = g.xl)
ORDER BY g.il, g.xl
