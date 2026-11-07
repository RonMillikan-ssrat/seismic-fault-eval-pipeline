-- 08 One summary row per survey: file metadata joined to trace statistics
--
-- Mechanics: broadcast join
--   surveys has one row per survey; trace_stats has one row per trace.
--   The BROADCAST hint ships a full copy of the small table to every executor,
--   so each task joins its own slice of the big table locally, with no shuffle.
--   Without the hint, Spark broadcasts automatically when the small side is under
--   spark.sql.autoBroadcastJoinThreshold (10 MB by default), when it can estimate
--   the size. Run with --explain 08 and look for BroadcastHashJoin in the plan,
--   rather than SortMergeJoin (which shuffles and sorts both sides).
SELECT /*+ BROADCAST(s) */
       s.survey, s.format_code, s.byteorder, s.ns, s.dt_ms, s.n_traces AS n_traces_file,
       COUNT(*)                                        AS n_traces_stats,
       SUM(CASE WHEN st.rms = 0 THEN 1 ELSE 0 END)     AS n_dead,
       ROUND(AVG(CASE WHEN st.rms > 0 THEN st.rms END), 2)          AS mean_rms_live,
       ROUND(AVG(CASE WHEN st.rms > 0 THEN st.dom_freq_hz END), 2)  AS mean_dom_freq_hz,
       ROUND(AVG(CASE WHEN st.rms > 0 THEN st.spec_centroid_hz END), 2) AS mean_centroid_hz
FROM trace_stats st
JOIN surveys s ON st.survey = s.survey
GROUP BY s.survey, s.format_code, s.byteorder, s.ns, s.dt_ms, s.n_traces
ORDER BY s.survey
