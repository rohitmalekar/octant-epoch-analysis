SELECT 
  events.project_id,
  bucket_month,
  event_type,
  sum(amount) as amount
FROM 
  `fleet-bongo-424111-b3.oso_production.int_events_monthly_to_project` AS events JOIN
  `fleet-bongo-424111-b3.oso_production.projects_by_collection_v1` AS collection ON
  events.project_id = collection.project_id
WHERE
  collection_name in ('octant-01','octant-02','octant-03','octant-04','octant-05')
  AND EXTRACT(YEAR FROM bucket_month) = 2024
GROUP BY 
  events.project_id, bucket_month, event_type
