SELECT 
  events.project_id,
  EXTRACT(quarter FROM bucket_day) AS quarter,
  COUNT(DISTINCT events.from_artifact_id) AS active_dev_count
FROM 
  `fleet-bongo-424111-b3.oso_production.int_events_daily_to_project` AS events JOIN
  `fleet-bongo-424111-b3.oso_production.projects_by_collection_v1` AS collection ON
  events.project_id = collection.project_id
WHERE
  collection_name in ('octant-01','octant-02','octant-03','octant-04','octant-05')
  AND events.event_type = 'COMMIT_CODE' 
  AND EXTRACT(YEAR FROM bucket_day) = 2024
GROUP BY 
  events.project_id, quarter
