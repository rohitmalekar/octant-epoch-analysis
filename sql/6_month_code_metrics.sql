SELECT distinct metrics.*
FROM `fleet-bongo-424111-b3.oso_production.code_metrics_by_project_v1` AS metrics JOIN
  `fleet-bongo-424111-b3.oso_production.projects_by_collection_v1` AS collection ON
  metrics.project_id = collection.project_id
where collection_name in ('octant-01','octant-02','octant-03','octant-04','octant-05')