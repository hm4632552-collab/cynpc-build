name: Build CyNPC
on:
  push:
    branches: [ main ]
  workflow_dispatch:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Get NDK
        run: |
          wget -q https://dl.google.com/android/repository/android-ndk-r27c-linux.zip
          unzip -q android-ndk-r27c-linux.zip
      - name: Get llama.cpp
        run: git clone --depth=1 https://github.com/ggerganov/llama.cpp llama
      - name: Build ARM64
        run: |
          cd llama
          cmake -B b64 -DCMAKE_TOOLCHAIN_FILE=../android-ndk-r27c/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=OFF
          cmake --build b64 -j$(nproc) --target llama-cli
          mkdir -p ../out64
          cp b64/bin/llama-cli ../out64/
          find b64 -name "*.so" -exec cp {} ../out64/ \; 2>/dev/null || true
      - name: Build ARM32
        run: |
          cd llama
          cmake -B b32 -DCMAKE_TOOLCHAIN_FILE=../android-ndk-r27c/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=android-26 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=OFF
          cmake --build b32 -j$(nproc) --target llama-cli
          mkdir -p ../out32
          cp b32/bin/llama-cli ../out32/
          find b32 -name "*.so" -exec cp {} ../out32/ \; 2>/dev/null || true
      - uses: actions/upload-artifact@v4
        with:
          name: llama-arm64
          path: out64/
      - uses: actions/upload-artifact@v4
        with:
          name: llama-arm32
          path: out32/
