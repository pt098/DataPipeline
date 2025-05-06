#!/bin/bash

# Script to build all Docker images for the data pipeline

echo "Creating output directory for visualizations..."
mkdir -p ../output

echo "Building Docker images..."

cd ..
docker-compose build

echo "Docker images built successfully." 