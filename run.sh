docker run -d --gpus all --cap-add=SYS_ADMIN --name llm-container --volume ncu-volume:/root sudo-accel-sim sleep infinity
#docker run -d --gpus all --cap-add=SYS_ADMIN --name llm-container sudo-accel-sim sleep infinity
docker exec -it llm-container /bin/bash
