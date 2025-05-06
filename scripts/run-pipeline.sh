#!/bin/bash

# Script to run the MPI data pipeline

cd ..
echo "Starting Docker containers..."
docker-compose up -d

echo "Waiting for containers to initialize..."
sleep 5

echo "Starting MPI data pipeline..."
docker exec controller-container bash -c "cd /app && mpirun --allow-run-as-root -np 5 -f rankfile.txt /app/data_pipeline"

echo "Pipeline execution completed."
echo "Results are available in the 'output' directory."

# Uncomment to automatically stop containers after completion
# echo "Stopping containers..."
# docker-compose down 