#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>

// Define our data structure for sensors
struct SensorData {
    int sensorId;
    double temperature;
    double humidity;
    double pressure;
    double timestamp;
    int status;
};

// Constants for the application
const int GENERATOR_RANK = 1;
const int PREPROCESSOR_RANK = 2;
const int ANALYZER_RANK = 3;
const int VISUALIZER_RANK = 4;
const int BATCH_SIZE = 100;
const int NUM_ITERATIONS = 20;
const int SHUTDOWN_SIGNAL = -1;

// Function to initialize MPI and get role information
std::string initializeMPI(int* argc, char*** argv, int* rank, int* numProcesses) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, numProcesses);
    
    char* roleEnv = std::getenv("ROLE");
    return (roleEnv != nullptr) ? std::string(roleEnv) : "unknown";
}

// Function to generate synthetic sensor data
std::vector<SensorData> generateData(int batchSize) {
    std::vector<SensorData> batch(batchSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Distributions for random values
    std::uniform_int_distribution<> sensorIdDist(1, 10);
    std::normal_distribution<> tempDist(25.0, 3.0);     // Mean 25°C, std dev 3
    std::normal_distribution<> humidityDist(50.0, 10.0); // Mean 50%, std dev 10
    std::normal_distribution<> pressureDist(1013.0, 5.0); // Mean 1013 hPa, std dev 5
    std::uniform_int_distribution<> statusDist(0, 3);     // 0: OK, 1: Warning, 2: Error, 3: Critical
    
    // Current timestamp
    double currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    
    for (int i = 0; i < batchSize; i++) {
        batch[i].sensorId = sensorIdDist(gen);
        batch[i].temperature = tempDist(gen);
        batch[i].humidity = humidityDist(gen);
        batch[i].pressure = pressureDist(gen);
        batch[i].timestamp = currentTime + i * 0.1;  // Add 0.1 seconds between readings
        batch[i].status = statusDist(gen);
    }
    
    return batch;
}

// Function to preprocess data (filter, transform)
std::vector<SensorData> preprocessData(const std::vector<SensorData>& rawData) {
    std::vector<SensorData> processedData;
    processedData.reserve(rawData.size());
    
    // Filter out invalid readings (example: temperature out of reasonable range)
    for (const auto& data : rawData) {
        if (data.temperature >= -30.0 && data.temperature <= 50.0 &&
            data.humidity >= 0.0 && data.humidity <= 100.0 &&
            data.pressure >= 900.0 && data.pressure <= 1100.0) {
            
            // Copy the data and apply transformations
            SensorData processed = data;
            
            // Convert temperature to Celsius if needed (assuming data already in Celsius)
            // processed.temperature = (data.temperature - 32.0) * 5.0 / 9.0;
            
            // Round values for consistency
            processed.temperature = std::round(processed.temperature * 10) / 10.0;
            processed.humidity = std::round(processed.humidity);
            processed.pressure = std::round(processed.pressure * 10) / 10.0;
            
            processedData.push_back(processed);
        }
    }
    
    return processedData;
}

// Structure to hold analysis results
struct AnalysisResults {
    double avgTemperature;
    double avgHumidity;
    double avgPressure;
    int errorCount;
    int warningCount;
    double minTemperature;
    double maxTemperature;
    double temperatureStdDev;
    int timestamp;
};

// Function to analyze preprocessed data
AnalysisResults analyzeData(const std::vector<SensorData>& data) {
    AnalysisResults results;
    if (data.empty()) {
        return results;
    }
    
    // Calculate averages
    double sumTemp = 0.0, sumHumidity = 0.0, sumPressure = 0.0;
    double minTemp = data[0].temperature;
    double maxTemp = data[0].temperature;
    int errorCount = 0, warningCount = 0;
    
    for (const auto& entry : data) {
        sumTemp += entry.temperature;
        sumHumidity += entry.humidity;
        sumPressure += entry.pressure;
        
        minTemp = std::min(minTemp, entry.temperature);
        maxTemp = std::max(maxTemp, entry.temperature);
        
        if (entry.status == 2 || entry.status == 3) {
            errorCount++;
        } else if (entry.status == 1) {
            warningCount++;
        }
    }
    
    results.avgTemperature = sumTemp / data.size();
    results.avgHumidity = sumHumidity / data.size();
    results.avgPressure = sumPressure / data.size();
    results.errorCount = errorCount;
    results.warningCount = warningCount;
    results.minTemperature = minTemp;
    results.maxTemperature = maxTemp;
    
    // Calculate standard deviation for temperature
    double sumSquaredDiff = 0.0;
    for (const auto& entry : data) {
        double diff = entry.temperature - results.avgTemperature;
        sumSquaredDiff += diff * diff;
    }
    results.temperatureStdDev = std::sqrt(sumSquaredDiff / data.size());
    
    // Current timestamp for the analysis
    results.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    return results;
}

// Function to save analysis results to a file (for visualization)
void saveResultsToFile(const AnalysisResults& results, int iteration) {
    std::ofstream outFile("/app/output/analysis_results.csv", 
                          iteration == 0 ? std::ios::out : std::ios::app);
    
    if (iteration == 0) {
        // Write header if first iteration
        outFile << "Timestamp,AvgTemp,MinTemp,MaxTemp,StdDevTemp,AvgHumidity,AvgPressure,Errors,Warnings\n";
    }
    
    outFile << results.timestamp << ","
            << std::fixed << std::setprecision(2) << results.avgTemperature << ","
            << results.minTemperature << ","
            << results.maxTemperature << ","
            << results.temperatureStdDev << ","
            << results.avgHumidity << ","
            << results.avgPressure << ","
            << results.errorCount << ","
            << results.warningCount << "\n";
            
    outFile.close();
}

// Controller process logic
void controllerProcess() {
    std::cout << "Controller started - managing data pipeline..." << std::endl;
    
    // Create the output directory if it doesn't exist
    system("mkdir -p /app/output");
    
    // Initialize counters for performance tracking
    int totalBatchesProcessed = 0;
    auto startTime = std::chrono::steady_clock::now();
    
    // Main controller loop
    int processedCount = 0;
    
    // Send startup signal to all nodes
    int signal = 1;
    MPI_Send(&signal, 1, MPI_INT, GENERATOR_RANK, 0, MPI_COMM_WORLD);
    
    // Wait for completion (only visualizer reports directly to controller)
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        AnalysisResults results;
        MPI_Recv(&results, sizeof(AnalysisResults), MPI_BYTE, VISUALIZER_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Iteration " << i+1 << "/" << NUM_ITERATIONS << " completed" << std::endl;
        std::cout << "  Avg Temperature: " << results.avgTemperature << " °C" << std::endl;
        std::cout << "  Avg Humidity: " << results.avgHumidity << " %" << std::endl;
        std::cout << "  Issues detected: " << results.errorCount << " errors, " 
                  << results.warningCount << " warnings" << std::endl;
        
        totalBatchesProcessed++;
    }
    
    // Send termination signal to all processes
    signal = SHUTDOWN_SIGNAL;
    MPI_Send(&signal, 1, MPI_INT, GENERATOR_RANK, 0, MPI_COMM_WORLD);
    
    // Calculate and display performance metrics
    auto endTime = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0;
    
    std::cout << "\nPipeline Performance:" << std::endl;
    std::cout << "Total processing time: " << totalTime << " seconds" << std::endl;
    std::cout << "Batches processed: " << totalBatchesProcessed << std::endl;
    std::cout << "Throughput: " << totalBatchesProcessed * BATCH_SIZE / totalTime << " records/sec" << std::endl;
    
    // Save performance data to file
    std::ofstream perfFile("/app/output/performance.txt");
    perfFile << "Pipeline Performance Summary" << std::endl;
    perfFile << "----------------------------" << std::endl;
    perfFile << "Total processing time: " << totalTime << " seconds" << std::endl;
    perfFile << "Batches processed: " << totalBatchesProcessed << std::endl;
    perfFile << "Records processed: " << totalBatchesProcessed * BATCH_SIZE << std::endl;
    perfFile << "Throughput: " << totalBatchesProcessed * BATCH_SIZE / totalTime << " records/sec" << std::endl;
    perfFile.close();
    
    std::cout << "Data pipeline completed successfully." << std::endl;
}

// Data generator process logic
void generatorProcess() {
    std::cout << "Data Generator started" << std::endl;
    
    int signal;
    MPI_Recv(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    int iteration = 0;
    while (signal != SHUTDOWN_SIGNAL && iteration < NUM_ITERATIONS) {
        // Generate random data
        std::vector<SensorData> batch = generateData(BATCH_SIZE);
        
        // Print status
        std::cout << "Generator: Produced batch " << iteration + 1 
                  << " with " << batch.size() << " records" << std::endl;
        
        // Send to preprocessor
        // First send the size
        int batchSize = batch.size();
        MPI_Send(&batchSize, 1, MPI_INT, PREPROCESSOR_RANK, 0, MPI_COMM_WORLD);
        
        // Then send the data
        MPI_Send(batch.data(), batchSize * sizeof(SensorData), MPI_BYTE, 
                 PREPROCESSOR_RANK, 0, MPI_COMM_WORLD);
        
        // Wait a bit to simulate real-time data generation
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        iteration++;
        
        // Check for signal from controller if we should continue
        int flag;
        MPI_Status status;
        MPI_Iprobe(0, 0, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            MPI_Recv(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Send shutdown signal to the next stage
    int shutdownSignal = SHUTDOWN_SIGNAL;
    MPI_Send(&shutdownSignal, 1, MPI_INT, PREPROCESSOR_RANK, 0, MPI_COMM_WORLD);
    
    std::cout << "Generator: Shutting down after " << iteration << " iterations" << std::endl;
}

// Data preprocessor logic
void preprocessorProcess() {
    std::cout << "Data Preprocessor started" << std::endl;
    
    bool running = true;
    int batchSize;
    int batches = 0;
    
    while (running) {
        // Receive batch size
        MPI_Recv(&batchSize, 1, MPI_INT, GENERATOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Check for shutdown signal
        if (batchSize == SHUTDOWN_SIGNAL) {
            running = false;
            break;
        }
        
        // Receive data batch
        std::vector<SensorData> rawData(batchSize);
        MPI_Recv(rawData.data(), batchSize * sizeof(SensorData), MPI_BYTE, 
                 GENERATOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Process the data
        std::vector<SensorData> processedData = preprocessData(rawData);
        
        batches++;
        std::cout << "Preprocessor: Processed batch " << batches 
                  << " - Input: " << rawData.size() << " records, Output: " 
                  << processedData.size() << " records" << std::endl;
        
        // Send processed data to analyzer
        batchSize = processedData.size();
        MPI_Send(&batchSize, 1, MPI_INT, ANALYZER_RANK, 0, MPI_COMM_WORLD);
        MPI_Send(processedData.data(), batchSize * sizeof(SensorData), MPI_BYTE, 
                 ANALYZER_RANK, 0, MPI_COMM_WORLD);
    }
    
    // Forward shutdown signal
    int shutdownSignal = SHUTDOWN_SIGNAL;
    MPI_Send(&shutdownSignal, 1, MPI_INT, ANALYZER_RANK, 0, MPI_COMM_WORLD);
    
    std::cout << "Preprocessor: Shutting down after processing " << batches << " batches" << std::endl;
}

// Data analyzer logic
void analyzerProcess() {
    std::cout << "Data Analyzer started" << std::endl;
    
    bool running = true;
    int batchSize;
    int batches = 0;
    
    while (running) {
        // Receive batch size
        MPI_Recv(&batchSize, 1, MPI_INT, PREPROCESSOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Check for shutdown signal
        if (batchSize == SHUTDOWN_SIGNAL) {
            running = false;
            break;
        }
        
        // Receive processed data
        std::vector<SensorData> data(batchSize);
        MPI_Recv(data.data(), batchSize * sizeof(SensorData), MPI_BYTE, 
                 PREPROCESSOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Analyze the data
        AnalysisResults results = analyzeData(data);
        
        batches++;
        std::cout << "Analyzer: Analyzed batch " << batches 
                  << " - " << data.size() << " records, Avg Temp: " 
                  << results.avgTemperature << "°C" << std::endl;
        
        // Send analysis results to visualizer
        MPI_Send(&results, sizeof(AnalysisResults), MPI_BYTE, VISUALIZER_RANK, 0, MPI_COMM_WORLD);
    }
    
    // Forward shutdown signal
    int shutdownSignal = SHUTDOWN_SIGNAL;
    MPI_Send(&shutdownSignal, 1, MPI_INT, VISUALIZER_RANK, 0, MPI_COMM_WORLD);
    
    std::cout << "Analyzer: Shutting down after analyzing " << batches << " batches" << std::endl;
}

// Data visualizer logic
void visualizerProcess() {
    std::cout << "Data Visualizer started" << std::endl;
    
    // Create the output directory
    system("mkdir -p /app/output");
    
    bool running = true;
    int iterations = 0;
    
    while (running) {
        // Receive analysis results
        AnalysisResults results;
        MPI_Status status;
        MPI_Recv(&results, sizeof(AnalysisResults), MPI_BYTE, ANALYZER_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Check message tag to see if it's a shutdown signal
        if (results.timestamp == 0 && results.avgTemperature == 0 && 
            results.avgHumidity == 0 && results.avgPressure == 0) {
            running = false;
            break;
        }
        
        // Save results to file for visualization
        saveResultsToFile(results, iterations);
        
        iterations++;
        std::cout << "Visualizer: Processed result " << iterations 
                  << " and saved to file" << std::endl;
        
        // Forward results to controller for monitoring
        MPI_Send(&results, sizeof(AnalysisResults), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    
    // Generate simple text-based visualization if we have data
    if (iterations > 0) {
        std::cout << "Visualizer: Creating final visualization" << std::endl;
        
        // Here we'd call our Python script for visualization
        system("python3 /app/visualize.py /app/output/analysis_results.csv /app/output/visualization.png");
    }
    
    std::cout << "Visualizer: Shutting down after processing " << iterations << " result sets" << std::endl;
}

int main(int argc, char** argv) {
    int rank, numProcesses;
    std::string role = initializeMPI(&argc, &argv, &rank, &numProcesses);
    
    // Set random seed based on rank for better randomness
    std::srand(std::time(nullptr) + rank);
    
    if (numProcesses < 5) {
        if (rank == 0) {
            std::cerr << "This program requires at least 5 processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    std::cout << "Process " << rank << " started with role: " << role << std::endl;
    
    // Execute appropriate function based on process rank
    try {
        switch (rank) {
            case 0: // Controller
                controllerProcess();
                break;
            case 1: // Generator
                generatorProcess();
                break;
            case 2: // Preprocessor
                preprocessorProcess();
                break;
            case 3: // Analyzer
                analyzerProcess();
                break;
            case 4: // Visualizer
                visualizerProcess();
                break;
            default:
                std::cout << "Process " << rank << " has no specific role assigned" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in process " << rank << ": " << e.what() << std::endl;
    }
    
    MPI_Finalize();
    return 0;
} 