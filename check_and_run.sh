#!/bin/bash
cd /home/molk/ml_project
source venv/bin/activate

# Chercher les mises à jour sur GitHub sans les fusionner
git fetch origin main

# Comparer la branche locale avec la branche distante
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ $LOCAL != $REMOTE ]; then
    echo "Changement détecté sur Git ! Lancement du pipeline..."
    git pull origin main
    # Déclenche le flow Prefect (via le CLI ou le déploiement)
    prefect deployment run 'ml-pipeline-all/ml-pipeline-all'
else
    echo "Aucun changement détecté."
fi