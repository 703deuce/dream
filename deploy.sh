#!/bin/bash

# DreamBooth Flux RunPod Serverless Deployment Script

set -e

# Configuration
IMAGE_NAME="dreambooth-flux"
REGISTRY="ghcr.io/703deuce"  # GitHub Container Registry
TAG="latest"

echo "🚀 DreamBooth Flux RunPod Serverless Deployment"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Docker build failed!"
    exit 1
fi

# Tag for registry
echo "🏷️  Tagging image for registry..."
docker tag ${IMAGE_NAME}:${TAG} ${REGISTRY}/${IMAGE_NAME}:${TAG}

# Push to registry
echo "📤 Pushing image to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}

if [ $? -eq 0 ]; then
    echo "✅ Image pushed successfully!"
    echo ""
    echo "🎉 Deployment completed!"
    echo ""
    echo "Next steps:"
    echo "1. Go to RunPod Console (https://runpod.io/console)"
    echo "2. Create a new Serverless endpoint"
    echo "3. Use this container image: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
    echo "4. Configure GPU: A100 (40GB) or H100 (80GB) recommended"
    echo "5. Set memory: 32GB minimum, 64GB recommended"
    echo "6. Set handler: handler.py"
    echo ""
    echo "📋 RunPod Configuration:"
    echo "   - Container Image: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
    echo "   - Handler: handler.py"
    echo "   - GPU: A100 or H100"
    echo "   - Memory: 32-64GB"
    echo "   - Volume: None (ephemeral storage)"
    echo ""
    echo "🔗 Test your endpoint with: python test_api.py"
else
    echo "❌ Failed to push image to registry!"
    echo "Please check your registry credentials and try again."
    exit 1
fi
