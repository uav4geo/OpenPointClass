FROM ubuntu:22.04 as builder

# Prerequisites
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt install -y --fix-missing --no-install-recommends \
    build-essential \
    software-properties-common \
    cmake \
    libtbb-dev \
    libboost-system-dev \
    libboost-serialization-dev \
    libpdal-dev \
    libeigen3-dev

COPY . /opc

# Run build
RUN cd /opc && mkdir build && cd build && cmake .. && make -j$(nproc)

FROM ubuntu:22.04 as runner

LABEL Author="Luca Di Leo <ldileo@digipa.it>" \
      Description="Fast and memory efficient semantic segmentation of 3D point clouds"

# Prerequisites
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    libtbb2 \
    libboost-iostreams1.74.0 \
    libboost-system1.74.0 \
    libboost-serialization1.74.0 \
    libpdal-base13 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get autoremove

COPY --from=builder /opc/build ./build

# Let pcclassify and pctrain be called without ./
ENV PATH="/build:$PATH"

WORKDIR /build



