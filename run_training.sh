#!/bin/bash

REPO_URL="https://github.com/osiastossou/YoloPHDWork.git"
REPO_DIR="YoloPHDWork"
LOG_FILE="log_osias.txt"

echo "=== DÉMARRAGE DU SCRIPT ==="

# 1. Vérifier si le dossier existe
if [ -d "$REPO_DIR" ]; then
    echo "Le dossier $REPO_DIR existe déjà."
    cd "$REPO_DIR" || exit 1
    echo "Mise à jour du dépôt (git pull)..."
    git pull origin main
else
    echo "Clonage du dépôt Git..."
    git clone "$REPO_URL"
    cd "$REPO_DIR" || exit 1
fi

# 2. Lancer l'entraînement et rediriger les logs
echo "Lancement de l'entraînement..."
python train.py >> "$LOG_FILE" 2>&1

echo "=== FIN DU SCRIPT ==="