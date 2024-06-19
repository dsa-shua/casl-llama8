#!/bin/bash

# Run this pls.

# For Profiling (Figure 4C)
# p_batch_sizes=(1 2 4 8 16 32 64 128 256)
# p_input_sizes=(1 2 4) # Lets keep it short for now.
# p_output_sizes=(64 128) # Doing shorter outputs does not make sense.


p_batch_sizes=(1 2 4 8 16 32 64 128 256)
p_input_sizes=(1) # Lets keep it short for now.
p_output_sizes=(256) # Doing shorter outputs does not make sense.

# Figure 2
t_batch_sizes=(1 2 4 8 16)
t_input_sizes=(2 4 8 16 32 128 256)
t_output_sizes=(2 4 8 16 32 128 256)


# if no arguments given, run everything.
if [ $# -ne 1 ]; then
    echo "Running LLAMA with Profiling"
    echo "This will generate {all, attn, mlp, blk}.txt files"
    echo "They should all be organized and can be found in /root/casl/llama/results/"

    # Pre-run: Remove results directory.
    if [ -d "results" ]; then
        # Clear everything.
        rm -rf results/
    fi

    echo "Rebuilding results/ directory"
    echo ""
    mkdir results/


    # PART 1: Generating all.txt 

    echo "If this is the first time running LLAMA3 8B, the first run"
    echo "will download 4 shards of the model, each about 5 GB each."
    echo ""

    # Make sure modeling_llama.py is not modified.
    rm -rf /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
    cp /root/casl/llama/include/base.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
    
    for b_size in "${p_batch_sizes[@]}"; do
        mkdir -p results/B"$b_size"/

        for o_size in "${p_output_sizes[@]}"; do
            mkdir -p results/B"$b_size"/O"$o_size"/

            for i_size in "${p_input_sizes[@]}"; do
                mkdir -p results/B"$b_size"/O"$o_size"/I"$i_size"/

                echo ""
                echo "Running profile_all.py (Batch: "$b_size" | OUT: "$o_size" | IN: "$i_size")"
                echo "Will generate all.txt"
                echo ""

                # Make sure that python llama script has the profilers
                python3 ./bin/profile_all.py -B "$b_size" -O "$o_size" -I "$i_size"
                
            done
        done
    done	      

    # PART 2: Generate mlp.txt and attn.txt

    # Make sure modeling_llama.py is modified to include profilers
    rm -rf /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
    cp /root/casl/llama/include/attn_mlp.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py

    for b_size in "${p_batch_sizes[@]}"; do
        for o_size in "${p_output_sizes[@]}"; do
            for i_size in "${p_input_sizes[@]}"; do
                echo ""
                echo "Running llama.py (Batch: "$b_size" | OUT: "$o_size" | IN: "$i_size")"
                echo "Will generate attn.txt and mlp.txt"
                echo ""


                # Make sure that llama main script does not have profilers
                python3 ./bin/llama.py -B "$b_size" -O "$o_size" -I "$i_size"

                # Move to respective directories
                mv ./results/attn.txt ./results/B"$b_size"/O"$o_size"/I"$i_size"/attn.txt
                mv ./results/mlp.txt ./results/B"$b_size"/O"$o_size"/I"$i_size"/mlp.txt

            done
        done
    done
    
    # Reset to base
    rm -rf /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
    cp /root/casl/llama/include/base.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py

    # PART3: Data Processing and Figure Generation

    echo ""
    echo "Running ./util/process.py to generate graphs"

    # Generate directories for the files
    mkdir ./results/img ./results/data
    for o_size in "${p_output_sizes[@]}"; do
        mkdir ./results/data/O"$o_size" ./results/img/O"$o_size"
        for i_size in "${p_input_sizes[@]}"; do
            mkdir ./results/data/O"$o_size"/I"$i_size" ./results/img/O"$o_size"/I"$i_size"
        done
    done
    python3 ./util/process.py



else
    case "$1" in
        raw)
            echo "Running base llama. No profiling"
            ./bin/llama.py
            ;;
        ones)
            echo "Running ones llama. No profiling"
            ./bin/llama.py -B 1 -O 1 -I 1
            ;;
        help)
            echo "To run with profilers, run: ./run.sh"
            echo "  But you can also run with other stuff: ./run.sh  <arg>"
            echo ""
            echo "  ARG LIST:"
            echo "      raw     -   run vanilla llama3 8B with no profiling"
            echo "      ones    -   llama with batch size 1, input and output len of 1"
            echo "      help    -   print this output."
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;

    esac

fi 
