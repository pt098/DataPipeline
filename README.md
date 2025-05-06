# Real-time Distributed Data Pipeline

## Project Overview
This project implements a real-time distributed data pipeline using MPI (Message Passing Interface) and Docker. The system processes streaming data through multiple stages running in separate containers.

## Architecture
The pipeline consists of four main components:

1. **Data Generator**: Produces synthetic data or ingests data from external sources
2. **Data Preprocessor**: Cleans, transforms, and prepares the raw data
3. **Data Analyzer**: Performs calculations, analysis and extracts insights
4. **Data Visualizer**: Collects results and prepares them for visualization

## Requirements
- Docker and Docker Compose
- OpenMPI
- C++ compiler with C++17 support
- Oracle VirtualBox (for deployment)

## Setup Instructions
1. Clone this repository
2. Generate SSH keys: `cd scripts && ./generate-keys.sh`
3. Build Docker images: `./build-images.sh`
4. Run the pipeline: `./run-pipeline.sh`

## Project Structure
- `src/`: Contains the C++ source code for all components
- `scripts/`: Shell scripts for building and running the system
- `keys/`: SSH keys for container communication
- `Dockerfile.*`: Docker configuration for each component

## Performance Metrics
The pipeline includes performance monitoring for:
- Throughput (records/second)
- Processing time per stage
- Resource utilization
- System scalability 