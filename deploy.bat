@echo off
REM DreamBooth Flux RunPod Serverless Deployment Script for Windows

setlocal enabledelayedexpansion

REM Configuration
set IMAGE_NAME=dreambooth-flux
set REGISTRY=ghcr.io/703deuce
set TAG=latest

echo 🚀 DreamBooth Flux RunPod Serverless Deployment
echo ================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Build the Docker image
echo 📦 Building Docker image...
docker build -t %IMAGE_NAME%:%TAG% .

if errorlevel 1 (
    echo ❌ Docker build failed!
    exit /b 1
) else (
    echo ✅ Docker image built successfully!
)

REM Tag for registry
echo 🏷️  Tagging image for registry...
docker tag %IMAGE_NAME%:%TAG% %REGISTRY%/%IMAGE_NAME%:%TAG%

REM Push to registry
echo 📤 Pushing image to registry...
docker push %REGISTRY%/%IMAGE_NAME%:%TAG%

if errorlevel 1 (
    echo ❌ Failed to push image to registry!
    echo Please check your registry credentials and try again.
    exit /b 1
) else (
    echo ✅ Image pushed successfully!
    echo.
    echo 🎉 Deployment completed!
    echo.
    echo Next steps:
    echo 1. Go to RunPod Console ^(https://runpod.io/console^)
    echo 2. Create a new Serverless endpoint
    echo 3. Use this container image: %REGISTRY%/%IMAGE_NAME%:%TAG%
    echo 4. Configure GPU: A100 ^(40GB^) or H100 ^(80GB^) recommended
    echo 5. Set memory: 32GB minimum, 64GB recommended
    echo 6. Set handler: handler.py
    echo.
    echo 📋 RunPod Configuration:
    echo    - Container Image: %REGISTRY%/%IMAGE_NAME%:%TAG%
    echo    - Handler: handler.py
    echo    - GPU: A100 or H100
    echo    - Memory: 32-64GB
    echo    - Volume: None ^(ephemeral storage^)
    echo.
    echo 🔗 Test your endpoint with: python test_api.py
)

pause



