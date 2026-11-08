-- 08 One summary row per survey (BigQuery)
-- Difference from the Spark version: there is no BROADCAST hint. BigQuery's
-- planner chooses between a broadcast join and a hash (shuffle) join itself; the
-- job's Execution details in the console show which one it used.
SELECT s.survey, s.format_code, s.byteorder, s.ns, s.dt_ms, s.n_traces AS n_traces_file,
       COUNT(*)                                       AS n_traces_stats,
       COUNTIF(st.rms = 0)                            AS n_dead,
       ROUND(AVG(IF(st.rms > 0, st.rms, NULL)), 2)              AS mean_rms_live,
       ROUND(AVG(IF(st.rms > 0, st.dom_freq_hz, NULL)), 2)      AS mean_dom_freq_hz,
       ROUND(AVG(IF(st.rms > 0, st.spec_centroid_hz, NULL)), 2) AS mean_centroid_hz
FROM `{ds}.trace_stats` st
JOIN `{ds}.surveys` s ON st.survey = s.survey
GROUP BY s.survey, s.format_code, s.byteorder, s.ns, s.dt_ms, s.n_traces
ORDER BY s.survey
