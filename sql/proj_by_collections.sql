SELECT * 
FROM `fleet-bongo-424111-b3.oso_production.projects_by_collection_v1`
where collection_name in ('octant-01','octant-02','octant-03','octant-04','octant-05')
order by collection_name, project_name

