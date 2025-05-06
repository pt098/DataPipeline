#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def create_visualizations(input_file, output_file):
    # Load the data
    try:
        data = pd.read_csv(input_file)
        
        # Convert timestamp to datetime
        data['Datetime'] = pd.to_datetime(data['Timestamp'], unit='s')
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(12, 10))
        
        # Temperature plot
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(data['Datetime'], data['AvgTemp'], 'b-', linewidth=2, label='Average')
        ax1.fill_between(data['Datetime'], data['MinTemp'], data['MaxTemp'], 
                         color='b', alpha=0.2, label='Min-Max Range')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Trends')
        ax1.grid(True)
        ax1.legend()
        
        # Humidity and Pressure plot
        ax2 = fig.add_subplot(3, 1, 2)
        line1 = ax2.plot(data['Datetime'], data['AvgHumidity'], 'g-', label='Humidity (%)')
        ax2.set_ylabel('Humidity (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        
        # Secondary y-axis for pressure
        ax2b = ax2.twinx()
        line2 = ax2b.plot(data['Datetime'], data['AvgPressure'], 'r-', label='Pressure (hPa)')
        ax2b.set_ylabel('Pressure (hPa)')
        ax2b.set_ylim(990, 1030)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        ax2.set_title('Humidity and Pressure')
        
        # Error/Warning count plot
        ax3 = fig.add_subplot(3, 1, 3)
        width = 0.35
        x = np.arange(len(data['Datetime']))
        ax3.bar(x - width/2, data['Errors'], width, label='Errors', color='red', alpha=0.7)
        ax3.bar(x + width/2, data['Warnings'], width, label='Warnings', color='orange', alpha=0.7)
        ax3.set_ylabel('Count')
        ax3.set_title('Anomaly Detection')
        ax3.set_xticks(x)
        ax3.set_xticklabels([dt.strftime('%H:%M:%S') for dt in data['Datetime']], rotation=45)
        ax3.legend()
        ax3.grid(True, axis='y')
        
        # Format x-axis dates for all subplots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.get_xticklabels(), rotation=45)
            
        plt.tight_layout()
        plt.savefig(output_file)
        
        # Create a summary statistics report
        with open(output_file.replace('.png', '_summary.txt'), 'w') as f:
            f.write("Data Pipeline Analysis Summary\n")
            f.write("===============================\n\n")
            f.write(f"Total records processed: {len(data)}\n\n")
            
            f.write("Temperature Statistics:\n")
            f.write(f"  Average: {data['AvgTemp'].mean():.2f}°C\n")
            f.write(f"  Min: {data['MinTemp'].min():.2f}°C\n")
            f.write(f"  Max: {data['MaxTemp'].max():.2f}°C\n")
            f.write(f"  Standard Deviation: {data['StdDevTemp'].mean():.2f}°C\n\n")
            
            f.write("Humidity Statistics:\n")
            f.write(f"  Average: {data['AvgHumidity'].mean():.2f}%\n\n")
            
            f.write("Pressure Statistics:\n")
            f.write(f"  Average: {data['AvgPressure'].mean():.2f} hPa\n\n")
            
            f.write("Anomaly Statistics:\n")
            f.write(f"  Total Errors: {data['Errors'].sum()}\n")
            f.write(f"  Total Warnings: {data['Warnings'].sum()}\n")
            
        print(f"Visualizations saved to {output_file} and summary to {output_file.replace('.png', '_summary.txt')}")
        return True
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <input_csv_file> <output_image_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = create_visualizations(input_file, output_file)
    sys.exit(0 if success else 1) 