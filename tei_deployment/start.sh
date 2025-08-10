#!/bin/bash

# --- 配置区 ---
# TEI 镜像
TEI_IMAGE="ghcr.io/huggingface/text-embeddings-inference:89-1.7"
# 模型在容器内的路径 (这个通常不需要改)
MODEL_PATH_INSIDE_CONTAINER="/model"
# 主机上的数据目录 (相对于此脚本)
HOST_DATA_PATH="$(pwd)/data"
# 其他 TEI 参数
TEI_ARGS="--pooling last-token --max-client-batch-size 256 --max-batch-tokens 300000"
# ----------------

# 自动检测GPU数量
detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1
    else
        echo "0"
    fi
}

# --- 1. 获取 GPU 数量 ---
if [ -z "$1" ]; then
    MAX_GPU_COUNT=$(detect_gpu_count)
    echo "侦测到 $MAX_GPU_COUNT 个可用的 GPU."
    # 为 GPU 数量的输入也加上 -e
    read -e -p "您想要使用多少个 GPU? (默认: $MAX_GPU_COUNT): " GPU_COUNT
    GPU_COUNT=${GPU_COUNT:-$MAX_GPU_COUNT}
else
    GPU_COUNT=$1
fi

if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || [ "$GPU_COUNT" -le 0 ]; then
    echo "错误：请输入一个有效的正整数作为 GPU 数量。"
    exit 1
fi

# --- 2. 获取模型路径 (已启用Tab自动补全) ---
DEFAULT_MODEL_PATH="$(pwd)/fused_db1"
if [ -z "$2" ]; then
    # 在 read 命令前添加 -e 参数来启用 Readline (包括Tab补全)
    read -e -p "请输入您的模型在主机上的路径 (默认: ${DEFAULT_MODEL_PATH}): " HOST_MODEL_PATH
    HOST_MODEL_PATH=${HOST_MODEL_PATH:-$DEFAULT_MODEL_PATH}
else
    HOST_MODEL_PATH=$2
    echo "使用命令行参数指定的模型路径: $HOST_MODEL_PATH"
fi

# 验证模型路径是否存在
if [ ! -d "$HOST_MODEL_PATH" ]; then
    echo "错误：指定的模型路径 '$HOST_MODEL_PATH' 不存在或不是一个目录。"
    exit 1
fi
echo "将使用模型路径: $HOST_MODEL_PATH"


echo "准备为 $GPU_COUNT 个 GPU 启动 TEI 服务..."

# --- 3. 生成 NGINX 配置文件 ---
NGINX_CONF_PATH="./nginx/nginx.conf"
mkdir -p ./nginx
echo "生成 NGINX 配置文件于: $NGINX_CONF_PATH"

UPSTREAM_BLOCK="upstream tei_backend {\n    least_conn;"
for i in $(seq 0 $(($GPU_COUNT - 1))); do
    UPSTREAM_BLOCK+="    server tei-$i:80;\n"
done
UPSTREAM_BLOCK+="}"

SERVER_BLOCK="server {\n    listen 80;\n\n    location / {\n        proxy_pass http://tei_backend;\n        proxy_http_version 1.1;\n        proxy_set_header Upgrade \$http_upgrade;\n        proxy_set_header Connection 'upgrade';\n        proxy_set_header Host \$host;\n        proxy_cache_bypass \$http_upgrade;\n    }\n}"

echo -e "$UPSTREAM_BLOCK\n\n$SERVER_BLOCK" > $NGINX_CONF_PATH

# --- 4. 生成 Docker Compose Override 文件 ---
COMPOSE_OVERRIDE_PATH="./docker-compose.override.yml"
echo "生成 Docker Compose Override 文件于: $COMPOSE_OVERRIDE_PATH"

echo "version: '3.8'
services:" > $COMPOSE_OVERRIDE_PATH

for i in $(seq 0 $(($GPU_COUNT - 1))); do
  echo "  tei-$i:
    image: ${TEI_IMAGE}
    container_name: tei-instance-${i}
    runtime: nvidia
    command: --model-id ${MODEL_PATH_INSIDE_CONTAINER} ${TEI_ARGS}
    volumes:
      - ${HOST_DATA_PATH}:/data
      - ${HOST_MODEL_PATH}:${MODEL_PATH_INSIDE_CONTAINER}
    networks:
      - tei-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              device_ids: ['${i}']
    restart: unless-stopped
" >> $COMPOSE_OVERRIDE_PATH
done

# --- 5. 启动服务 ---
echo "使用 Docker Compose 启动所有服务..."
docker-compose -f docker-compose.yml -f $COMPOSE_OVERRIDE_PATH up -d --remove-orphans

echo ""
echo "部署完成!"
echo "服务已绑定到 0.0.0.0:8080，您可以从网络中的其他机器访问。"
echo "请使用运行此服务的主机的IP地址进行访问，例如："
echo "http://<YOUR_MACHINE_IP>:8080"
echo ""
echo "重要提示: 请确保您的主机防火墙允许外部访问 TCP 8080 端口。"
echo ""
echo "要停止服务, 请运行: docker-compose -f docker-compose.yml -f docker-compose.override.yml down"