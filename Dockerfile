FROM klue980/nvidia:trt-open-24.03.29


USER root

WORKDIR /root/casl/llama
RUN apt update && apt update -y

# COPY llama /root/casl/llama
ENV LIB_DIR=/usr/local/lib/python3.10/dist-packages/transformers/models/llama
ENV LLAMA_SOUCE=/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py

# Setup virtual environment (but not really used...)
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN cp /root/miniconda3/bin/activate /bin
RUN source activate

# Copy Source Files
COPY bin/ /root/casl/llama/bin
COPY script/run.sh  /root/casl/llama/
COPY util/ /root/casl/llama/util
COPY include/ /root/casl/llama/include

RUN chmod +x /root/casl/llama/run.sh
RUN chmod +x bin/*
RUN chmod +x util/*

RUN mv /root/casl/llama/include/README /root/casl/llama/README

CMD ["cat", "/root/casl/llama/README"]

