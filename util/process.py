#!/usr/bin/env python3


# WHAT: using {attn, mlp, all}.txt files, read the profiled CUDA times of the attention and mlp layers.
#   Generate a graph after.
# This should be ran once only!


import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# This should match the array from run.sh!
batch_list  = [1,2,4,8,16,32,64,128,256]
input_list  = [1]
output_list = [256]


for output_size in output_list:
    for input_size in input_list:
        csv_data = [["Batch Size", "Attention Time", "MLP Time", "Other"]]

        model = "LLAMA3 8B" 
        for batch_size in batch_list:
            
            cuda_file = f"/root/casl/llama/results/B{batch_size}/O{output_size}/I{input_size}/all.txt"
            attn_file = f"/root/casl/llama/results/B{batch_size}/O{output_size}/I{input_size}/attn.txt"
            ff_file   = f"/root/casl/llama/results/B{batch_size}/O{output_size}/I{input_size}/mlp.txt"

            TOTAL_FF_TIME = 0
            TOTAL_ATTN_TIME = 0
            TOTAL_CUDA_TIME = 0
            
            csv_line = []

            with open(cuda_file, "r") as t_file:
                lines = t_file.readlines()
                
                for line in lines:
                    line = line.rstrip("\n")
                    if "Self CUDA time total:" in line:
                        temp = line.split(":")
                        raw_time = temp[1]

                        # Normalize to ms
                        if raw_time[-2:] == "ms":
                            TOTAL_CUDA_TIME = float(raw_time[:-2])
                        elif raw_time[-2:] == "us":
                            TOTAL_CUDA_TIME = float(raw_time[:-2]) / 1000.0
                        else: # seconds
                            TOTAL_CUDA_TIME = float(raw_time[:-2]) * 1000.0


            with open(attn_file, "r") as t_file:
                lines = t_file.readlines()
                
                for line in lines:
                    line = line.rstrip("\n")
                    if "Self CUDA time total:" in line:
                        temp = line.split(":")
                        raw_time = temp[1]
                        
                        # Normalize to ms
                        if raw_time[-2:] == "ms":
                            TOTAL_ATTN_TIME += float(raw_time[:-2])
                        elif raw_time[-2:] == "us":
                            TOTAL_ATTN_TIME += float(raw_time[:-2]) / 1000.0
                        else: # seconds
                            TOTAL_ATTN_TIME += float(raw_time[:-2]) * 1000.0


            with open(ff_file, "r") as t_file:
                lines = t_file.readlines()
                
                for line in lines:
                    line = line.rstrip("\n")
                    if "Self CUDA time total:" in line:
                        temp = line.split(":")
                        raw_time = temp[1]
                        
                        # Normalize to ms
                        if raw_time[-2:] == "ms":
                            TOTAL_FF_TIME += float(raw_time[:-2])
                        elif raw_time[-2:] == "us":
                            TOTAL_FF_TIME += float(raw_time[:-2]) / 1000.0
                        else: # seconds
                            TOTAL_FF_TIME += float(raw_time[:-2]) * 1000.0


            other = TOTAL_CUDA_TIME - (TOTAL_ATTN_TIME + TOTAL_FF_TIME)
            csv_line = [batch_size, TOTAL_ATTN_TIME, TOTAL_FF_TIME, other]
            csv_data.append(csv_line)
            
            isAttentionBigger = TOTAL_ATTN_TIME > TOTAL_FF_TIME

            print("BATCH SIZE:                  ", batch_size)
            print("TOTAL CUDA TIME:             ",TOTAL_CUDA_TIME, "ms")
            print("ATTENTION TIME:              ",TOTAL_ATTN_TIME, "ms")
            print("MLP TIME:                    ",TOTAL_FF_TIME, "ms")
            print("IS ATTENION BIGGER?:         ",isAttentionBigger)
            print("SUM OF MLP AND ATTN TIME:    ", TOTAL_ATTN_TIME + TOTAL_FF_TIME, "ms")
            print("")
            
        # Collect Data to CSV.
        with open(f"/root/casl/llama/results/data/O{output_size}/I{input_size}/collect.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
            
        batch_sizes = []
        attention_times = []
        mlp_times = []
        other_times = []


        # Using generated CSV file, generate image.
        with open(f"/root/casl/llama/results/data/O{output_size}/I{input_size}/collect.csv", mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                batch_sizes.append(int(row[0]))
                attention_times.append(float(row[1]))
                mlp_times.append(float(row[2]))
                other_times.append(float(row[3]))

        # Create the x positions for the bars
        x_positions = np.arange(len(batch_sizes))

        # Plot the stacked bar chart
        plt.figure(figsize=(10, 6))

        # Create the bar chart
        mlp_bars = plt.bar(x_positions, mlp_times, label='MLP Time')
        attn_bars = plt.bar(x_positions, attention_times, bottom=mlp_times, label='Attention Time')
        other_bars = plt.bar(x_positions, other_times, bottom=np.array(attention_times) + np.array(mlp_times), label='ETC Time')

        plt.xlabel('Batch Size')
        plt.ylabel('Time (ms)')
        plt.title(f'{model} Model [I: {input_size} | O: {output_size}] Attention and MLP Latency')
        plt.xticks(x_positions, batch_sizes)
        plt.legend()
        plt.grid("minor", axis='y')

        plt.savefig(f"/root/casl/llama/results/img/O{output_size}/I{input_size}/figure.png")

        print(f"\n\nFigure Generated in: /root/casl/llama/results/img/O{output_size}/I{input_size}/figure.png\n")