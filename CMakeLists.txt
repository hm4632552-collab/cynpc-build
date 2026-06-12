name: Build CyNPC

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build-android:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Clone llama.cpp
      run: git clone --depth=1 https://github.com/ggerganov/llama.cpp llama

    - name: Copy our engine
      run: cp cynpc_engine.cpp llama/examples/cynpc_engine.cpp

    - name: Setup NDK
      run: |
        wget -q https://dl.google.com/android/repository/android-ndk-r27c-linux.zip
        unzip -q android-ndk-r27c-linux.zip
        echo "NDK=$PWD/android-ndk-r27c" >> $GITHUB_ENV

    - name: Build ARM64
      run: |
        cd llama
        cmake -B build64 \
          -DCMAKE_TOOLCHAIN_FILE=${{ env.NDK }}/build/cmake/android.toolchain.cmake \
          -DANDROID_ABI=arm64-v8a \
          -DANDROID_PLATFORM=android-26 \
          -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_BUILD_TESTS=OFF \
          -DLLAMA_BUILD_EXAMPLES=ON \
          -DGGML_NATIVE=OFF
        cmake --build build64 -j$(nproc) --target llama-cli
        mkdir -p ../out/arm64
        cp build64/bin/llama-cli ../out/arm64/
        find build64 -name "*.so" -exec cp {} ../out/arm64/ \; 2>/dev/null || true

    - name: Build ARM32
      run: |
        cd llama
        cmake -B build32 \
          -DCMAKE_TOOLCHAIN_FILE=${{ env.NDK }}/build/cmake/android.toolchain.cmake \
          -DANDROID_ABI=armeabi-v7a \
          -DANDROID_PLATFORM=android-26 \
          -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_BUILD_TESTS=OFF \
          -DLLAMA_BUILD_EXAMPLES=ON \
          -DGGML_NATIVE=OFF
        cmake --build build32 -j$(nproc) --target llama-cli
        mkdir -p ../out/arm32
        cp build32/bin/llama-cli ../out/arm32/
        find build32 -name "*.so" -exec cp {} ../out/arm32/ \; 2>/dev/null || true

    - name: Upload ARM64
      uses: actions/upload-artifact@v4
      with:
        name: llama-arm64
        path: out/arm64/
        retention-days: 30

    - name: Upload ARM32
      uses: actions/upload-artifact@v4
      with:
        name: llama-arm32
        path: out/arm32/
        retention-days: 30
