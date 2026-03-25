# Multi-stage Dockerfile for MeshParameterization
# Stage 1: Build C++ server + CLI tools
# Stage 2: Runtime with Python + Gmsh

# ============================================================
# Stage 1: Build
# ============================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates \
    libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Optional: Intel MKL for Composite Majorization method
# Uncomment to enable CM (adds ~500MB to image):
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     intel-oneapi-mkl-devel \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY server/ server/
COPY cgal_param/ cgal_param/
COPY extern/ extern/
COPY tests/CMakeLists.txt tests/
COPY tests/*.cpp tests/

# Build server + bench CLI
# FetchContent downloads Eigen, libigl, CGAL, Boost, tinygltf, Spectra
RUN cd server && \
    cmake -G Ninja -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O2" && \
    cmake --build build --parallel $(nproc) && \
    strip build/meshparam_server build/meshparam_bench 2>/dev/null || true

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies: Python, Gmsh, Node.js (for Vite dev server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgomp1 libglu1-mesa libegl1 libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Copy built binaries from builder
COPY --from=builder /build/server/build/meshparam_server /app/server/build/meshparam_server
COPY --from=builder /build/server/build/meshparam_bench /app/server/build/meshparam_bench

# Copy application files
COPY scripts/ /app/scripts/
COPY web/ /app/web/
COPY config.json /app/
COPY models/step/ /app/models/step/
COPY models/glTF/ /app/models/glTF/

# Install Node.js dependencies for frontend (optional, for Vite dev mode)
# For production, serve web/ as static files from the C++ server
RUN if [ -f /app/web/package.json ]; then \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y nodejs && \
        cd /app/web && npm install --production && \
        rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

EXPOSE 8080

# Run server with config file
CMD ["./server/build/meshparam_server", \
     "--port", "8080", \
     "--web-root", "web", \
     "--gmsh-cli", "python3 scripts/occ_gmsh_pipeline.py", \
     "--log", "/app/logs/meshparam.log"]
