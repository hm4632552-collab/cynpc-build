# ╔══════════════════════════════════════════════════════════════╗
# ║  CyNPC Engine v2.0 — CMakeLists.txt                         ║
# ║  يبني: libcynpc.so (library) + cynpc-cli (binary)           ║
# ╚══════════════════════════════════════════════════════════════╝

cmake_minimum_required(VERSION 3.22)
project(cynpc_engine VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── مسار llama.cpp (عدّله حسب مشروعك) ──────────────────────────
set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp"
    CACHE PATH "مسار مجلد llama.cpp")

# ── ARM تحسينات ──────────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "🔧 ARM64 — تفعيل NEON + crypto")
    add_compile_options(
        -march=armv8-a+crypto+crc
        -mtune=cortex-a55       # Snapdragon/MediaTek المنتشر
        -O3
        -ffast-math
        -funroll-loops
        -fomit-frame-pointer
        -fvisibility=hidden
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message(STATUS "🔧 ARM32 — تفعيل NEON")
    add_compile_options(
        -march=armv7-a
        -mfpu=neon-vfpv4
        -mfloat-abi=softfp
        -O3
        -ffast-math
        -funroll-loops
        -fomit-frame-pointer
        -fvisibility=hidden
    )
else()
    add_compile_options(-O2)
endif()

# ── Link Time Optimization ────────────────────────────────────────
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok)
if(ipo_ok)
    message(STATUS "✅ LTO مفعّل")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ── llama.cpp headers + libs ──────────────────────────────────────
include_directories(
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
)

# ══════════════════════════════════════════════════════════════════
# TARGET 1: libcynpc.so  — shared library للـ JNI
# ══════════════════════════════════════════════════════════════════
add_library(cynpc SHARED cynpc_engine.cpp)

target_compile_definitions(cynpc PRIVATE CYNPC_LIBRARY_ONLY)

target_link_libraries(cynpc PRIVATE
    ${LLAMA_DIR}/build/src/libllama.a
    ${LLAMA_DIR}/build/ggml/src/libggml.a
    ${LLAMA_DIR}/build/common/libcommon.a
    android
    log
    m
    dl
)

# شيّل الـ symbols غير الضرورية للتصغير
target_link_options(cynpc PRIVATE
    -Wl,--gc-sections
    -Wl,--strip-all
    -Wl,--exclude-libs,ALL
)

set_target_properties(cynpc PROPERTIES
    OUTPUT_NAME     "cynpc"
    SOVERSION       2
    VERSION         2.0.0
)

# ══════════════════════════════════════════════════════════════════
# TARGET 2: cynpc-cli  — binary للاختبار من Terminal
# ══════════════════════════════════════════════════════════════════
add_executable(cynpc-cli cynpc_engine.cpp)

target_link_libraries(cynpc-cli PRIVATE
    ${LLAMA_DIR}/build/src/libllama.a
    ${LLAMA_DIR}/build/ggml/src/libggml.a
    ${LLAMA_DIR}/build/common/libcommon.a
    android
    log
    m
    dl
)

target_link_options(cynpc-cli PRIVATE
    -Wl,--gc-sections
    -static-libstdc++      # لتشغيله بدون libc++_shared.so
)

# ══════════════════════════════════════════════════════════════════
# Install
# ══════════════════════════════════════════════════════════════════
install(TARGETS cynpc cynpc-cli
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

message(STATUS "
╔══════════════════════════════╗
║  CyNPC Engine v2.0 Ready     ║
║  ABI: ${CMAKE_SYSTEM_PROCESSOR}
║  LTO: ${ipo_ok}
╚══════════════════════════════╝")
