from pipeline_prefect import flow_all, flow_train
from prefect.deployments import Deployment
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta

# Déploiement du pipeline complet
deployment_all = Deployment.build_from_flow(
    flow=flow_all,
    name="ml-pipeline-all",
    schedules=[IntervalSchedule(interval=timedelta(days=1))],
    work_queue_name="default",
)

# Déploiement du pipeline d'entraînement seul
deployment_train = Deployment.build_from_flow(
    flow=flow_train,
    name="ml-pipeline-train",
    work_queue_name="default",
)

if __name__ == "__main__":
    deployment_all.apply()
    deployment_train.apply()
    print("Déploiements enregistrés avec succès sur le serveur Prefect !")