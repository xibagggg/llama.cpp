llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp LINK llama)
    llama_build_and_test(test-sampling.cpp LINK llama)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp LINK llama)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp LINK llama)
    llama_build_and_test(test-batch-alloc.cpp LINK llama)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    # TODO: after `test-llama-archs` is refactored to not link internal symbols, we can move all related tests out of
    #       this branch and run them on all platforms
    llama_build(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # Test context state handling across all architectures, using the generated dummy models
    llama_test(
        test-llama-context
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    # abort on graph reallocations that the initial reserve should have covered
    set_tests_properties(test-llama-context PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context NAME test-llama-context-model LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

llama_build(test-backend-ops.cpp LINK ggml)

llama_build_and_test(test-tiled-mulmat.cpp LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp LINK ggml)
    llama_build_and_test(test-quantize-perf.cpp LINK ggml)
    llama_build_and_test(test-rope.cpp LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp LINK ggml)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp LINK mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model" LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

# K3 numerical regression for the opt-in Paddle vision IME2 kernel.
if (GGML_SPACEMIT)
    llama_build(test-paddle-f16-ime2.cpp LINK ggml)
    set(paddle_f16_env "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0")
    set(paddle_f16_reference "${CMAKE_CURRENT_BINARY_DIR}/paddle-f16-ime2-panel-reference.bin")
    # A fresh process is needed for each panel count because the kernel caches
    # its environment setting. The fixture always regenerates the full output.
    llama_test(test-paddle-f16-ime2 LABEL spacemit ARGS --write-reference "${paddle_f16_reference}")
    set_tests_properties(test-paddle-f16-ime2 PROPERTIES
        ENVIRONMENT "${paddle_f16_env};GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=1;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=8;GGML_SPACEMIT_PADDLE_F16_M16=0"
        FIXTURES_SETUP paddle_f16_panel_reference RUN_SERIAL TRUE)
    foreach(paddle_f16_panels RANGE 2 4)
        set(paddle_f16_test "test-paddle-f16-ime2-panels-${paddle_f16_panels}")
        llama_test(test-paddle-f16-ime2 NAME "${paddle_f16_test}" LABEL spacemit
            ARGS --compare-reference "${paddle_f16_reference}")
        set_tests_properties("${paddle_f16_test}" PROPERTIES
            ENVIRONMENT "${paddle_f16_env};GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=${paddle_f16_panels};GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=8;GGML_SPACEMIT_PADDLE_F16_M16=0"
            FIXTURES_REQUIRED paddle_f16_panel_reference RUN_SERIAL TRUE)
    endforeach()
    # Wider activation packing and M16 scheduling must preserve the same bits.
    # K4304 retains its supported fallback when these switches are requested.
    foreach(paddle_f16_m16 RANGE 0 1)
        foreach(paddle_f16_panels IN ITEMS 1 4)
            set(paddle_f16_test "test-paddle-f16-ime2-pack64-m16-${paddle_f16_m16}-panels-${paddle_f16_panels}")
            llama_test(test-paddle-f16-ime2 NAME "${paddle_f16_test}" LABEL spacemit
                ARGS --compare-reference "${paddle_f16_reference}")
            set_tests_properties("${paddle_f16_test}" PROPERTIES
                ENVIRONMENT "${paddle_f16_env};GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=${paddle_f16_panels};GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=64;GGML_SPACEMIT_PADDLE_F16_M16=${paddle_f16_m16}"
                FIXTURES_REQUIRED paddle_f16_panel_reference RUN_SERIAL TRUE)
        endforeach()
    endforeach()
endif()

if (GGML_SPACEMIT)
    llama_build(test-paddle-attention.cpp LINK ggml)
    llama_test(test-paddle-attention LABEL spacemit)
    set(paddle_attention_env "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0")
    set_tests_properties(test-paddle-attention PROPERTIES ENVIRONMENT
        "${paddle_attention_env};GGML_SPACEMIT_OCR_FA_FULL_KV=0")
    add_test(NAME test-paddle-attention-fullkv-two-pass COMMAND $<TARGET_FILE:test-paddle-attention>)
    set_tests_properties(test-paddle-attention-fullkv-two-pass PROPERTIES ENVIRONMENT
        "${paddle_attention_env};GGML_SPACEMIT_OCR_FA_FULL_KV=2")
    # N64/65 must not alias an undersized probability buffer as K64 scratch.
    add_test(NAME test-paddle-attention-fullkv-head-cache COMMAND $<TARGET_FILE:test-paddle-attention>)
    set_tests_properties(test-paddle-attention-fullkv-head-cache PROPERTIES ENVIRONMENT
        "${paddle_attention_env};GGML_SPACEMIT_OCR_FA_FULL_KV=1;GGML_SPACEMIT_OCR_FA_HEAD_K_TCM=1;GGML_SPACEMIT_OCR_FA_FULL_PACK=1;GGML_SPACEMIT_OCR_FA_ALL_K_TCM=0")
    set_tests_properties(test-paddle-attention test-paddle-attention-fullkv-two-pass
        test-paddle-attention-fullkv-head-cache PROPERTIES LABELS spacemit RUN_SERIAL TRUE)
endif()


# CPU/AI elementwise regression for the opt-in VISION RoPE implementation.
if (GGML_SPACEMIT)
    llama_build(test-paddle-vision-rope.cpp LINK ggml)
    llama_test(test-paddle-vision-rope LABEL spacemit)
    set_tests_properties(test-paddle-vision-rope PROPERTIES RUN_SERIAL TRUE ENVIRONMENT
        "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0")
endif()

if (GGML_SPACEMIT)
    add_test(NAME test-paddle-vision-rope-no-cache COMMAND $<TARGET_FILE:test-paddle-vision-rope>)
    set_tests_properties(test-paddle-vision-rope-no-cache PROPERTIES RUN_SERIAL TRUE LABELS spacemit ENVIRONMENT
        "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0;GGML_SPACEMIT_VISION_ROPE_CACHE=0")
endif()

if (GGML_SPACEMIT)
    llama_test(test-paddle-f16-ime2 NAME test-paddle-f16-ime2-kblock-1024 LABEL spacemit
        ARGS --compare-reference "${paddle_f16_reference}")
    set_tests_properties(test-paddle-f16-ime2-kblock-1024 PROPERTIES
        ENVIRONMENT "${paddle_f16_env};GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=4;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=64;GGML_SPACEMIT_PADDLE_F16_M16=1;GGML_SPACEMIT_PADDLE_F16_KBLOCK=1024"
        FIXTURES_REQUIRED paddle_f16_panel_reference RUN_SERIAL TRUE)
endif()

# Direct CPU/SPACEMIT F32-to-F16 permutation/view byte regression.
if (GGML_SPACEMIT)
    llama_build(test-paddle-strided-cast.cpp LINK ggml)
    llama_test(test-paddle-strided-cast LABEL spacemit)
    set(paddle_strided_cast_env "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0;GGML_SPACEMIT_OCR_STRIDED_CAST=1")
    set_tests_properties(test-paddle-strided-cast PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "${paddle_strided_cast_env}")
endif()

# Independent F16 GEMM+bias regression: trace-proven positive/negative matches,
# every output byte, changed activations/biases and real vision geometries.
if (GGML_SPACEMIT)
    llama_build(test-paddle-f16-bias.cpp LINK ggml)
    set(paddle_f16_bias_reference "${CMAKE_CURRENT_BINARY_DIR}/paddle-f16-bias-reference.bin")
    set(paddle_f16_bias_env "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0")
    llama_test(test-paddle-f16-bias NAME test-paddle-f16-bias-reference LABEL spacemit
        ARGS --write-reference "${paddle_f16_bias_reference}")
    set_tests_properties(test-paddle-f16-bias-reference PROPERTIES
        ENVIRONMENT "${paddle_f16_bias_env};GGML_SPACEMIT_PADDLE_F16_BIAS_FUSE=0;GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=1;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=8;GGML_SPACEMIT_PADDLE_F16_M16=0;GGML_SPACEMIT_PADDLE_F16_KBLOCK=0"
        FIXTURES_SETUP paddle_f16_bias_reference RUN_SERIAL TRUE)
    foreach(paddle_f16_bias_mode IN ITEMS base m16 kblock1024)
        set(paddle_f16_bias_options "GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=1;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=8;GGML_SPACEMIT_PADDLE_F16_M16=0;GGML_SPACEMIT_PADDLE_F16_KBLOCK=0")
        if (NOT paddle_f16_bias_mode STREQUAL "base")
            set(paddle_f16_bias_options "GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=4;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=64;GGML_SPACEMIT_PADDLE_F16_M16=1;GGML_SPACEMIT_PADDLE_F16_KBLOCK=0")
        endif()
        if (paddle_f16_bias_mode STREQUAL "kblock1024")
            set(paddle_f16_bias_options "GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=4;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=64;GGML_SPACEMIT_PADDLE_F16_M16=1;GGML_SPACEMIT_PADDLE_F16_KBLOCK=1024")
        endif()
        set(paddle_f16_bias_test "test-paddle-f16-bias-${paddle_f16_bias_mode}")
        llama_test(test-paddle-f16-bias NAME "${paddle_f16_bias_test}" LABEL spacemit
            ARGS --compare-reference "${paddle_f16_bias_reference}")
        set_tests_properties("${paddle_f16_bias_test}" PROPERTIES
            ENVIRONMENT "${paddle_f16_bias_env};GGML_SPACEMIT_PADDLE_F16_BIAS_FUSE=1;${paddle_f16_bias_options}"
            FIXTURES_REQUIRED paddle_f16_bias_reference RUN_SERIAL TRUE)
    endforeach()
endif()

# Same-backend F32 byte regression for the independently selected elementwise fusions.
if (GGML_SPACEMIT)
    llama_build(test-paddle-elementwise-fusion.cpp LINK ggml)
    set(paddle_elementwise_reference "${CMAKE_CURRENT_BINARY_DIR}/paddle-elementwise-fusion-reference.bin")
    llama_test(test-paddle-elementwise-fusion LABEL spacemit ARGS --write-reference "${paddle_elementwise_reference}")
    set_tests_properties(test-paddle-elementwise-fusion PROPERTIES RUN_SERIAL TRUE
        ENVIRONMENT "${paddle_f16_env};GGML_SPACEMIT_NORM_AFFINE=0;GGML_SPACEMIT_ADD_GELU=0"
        FIXTURES_SETUP paddle_elementwise_reference)
    foreach(paddle_elementwise_pair IN ITEMS "1,0" "0,1" "1,1")
        string(REPLACE "," ";" paddle_elementwise_values "${paddle_elementwise_pair}")
        list(GET paddle_elementwise_values 0 paddle_elementwise_norm)
        list(GET paddle_elementwise_values 1 paddle_elementwise_gelu)
        set(paddle_elementwise_name "test-paddle-elementwise-fusion-n${paddle_elementwise_norm}-g${paddle_elementwise_gelu}")
        llama_test(test-paddle-elementwise-fusion NAME "${paddle_elementwise_name}" LABEL spacemit
            ARGS --compare-reference "${paddle_elementwise_reference}")
        set_tests_properties("${paddle_elementwise_name}" PROPERTIES RUN_SERIAL TRUE
            ENVIRONMENT "${paddle_f16_env};GGML_SPACEMIT_NORM_AFFINE=${paddle_elementwise_norm};GGML_SPACEMIT_ADD_GELU=${paddle_elementwise_gelu}"
            FIXTURES_REQUIRED paddle_elementwise_reference)
    endforeach()
endif()

# Narrow position interpolation: actual CPU/SPACEMIT byte comparison and off gate.
if (GGML_SPACEMIT)
    llama_build(test-paddle-pos-resize.cpp LINK ggml)
    llama_test(test-paddle-pos-resize LABEL spacemit)
    add_test(NAME test-paddle-pos-resize-disabled COMMAND $<TARGET_FILE:test-paddle-pos-resize> --disabled)
    set(paddle_pos_resize_env "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0")
    set_tests_properties(test-paddle-pos-resize test-paddle-pos-resize-disabled PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "${paddle_pos_resize_env}")
endif()


# Append after the existing Paddle attention tests in tests/CMakeLists.txt.
# Source only; no backend/CMake/source file was modified by this candidate.
if (GGML_SPACEMIT AND UNIX AND NOT APPLE)
    llama_build(test-paddle-attention-pack.cpp LINK ggml)

    # Each configure creates a NEW immutable reference path. The executable
    # refuses to replace an existing file. Reconfigure before rerunning the
    # fixture setup, or set a fresh PADDLE_ATTENTION_PACK_RUN_ID explicitly.
    if (NOT DEFINED PADDLE_ATTENTION_PACK_RUN_ID)
        string(TIMESTAMP paddle_attention_pack_timestamp "%Y%m%d-%H%M%S" UTC)
        string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef paddle_attention_pack_nonce)
        set(PADDLE_ATTENTION_PACK_RUN_ID "${paddle_attention_pack_timestamp}-${paddle_attention_pack_nonce}")
    endif()
    set(paddle_attention_pack_reference
        "${CMAKE_CURRENT_BINARY_DIR}/paddle-attention-pack-${PADDLE_ATTENTION_PACK_RUN_ID}.bin")
    set(paddle_attention_pack_env
        "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0")

    # Six-bit order is FULL_PACK, LEAN_PREPARE, FUSED_QUERY, HALF_PACK,
    # SOFTMAX_PACK (only 0/1), COMPACT. Every test owns a fresh process.
    # The executable fixes FULL_KV=2 and unrelated experimental flags itself.
    function(paddle_attention_pack_add name bits write_reference)
        if (write_reference)
            set(paddle_attention_pack_action --write-reference)
        else()
            set(paddle_attention_pack_action --compare-reference)
        endif()
        set(paddle_attention_pack_options "${paddle_attention_pack_env}")
        set(paddle_attention_pack_names FULL_PACK LEAN_PREPARE FUSED_QUERY HALF_PACK SOFTMAX_PACK COMPACT)
        foreach(paddle_attention_pack_index RANGE 0 5)
            string(SUBSTRING "${bits}" ${paddle_attention_pack_index} 1 paddle_attention_pack_value)
            list(GET paddle_attention_pack_names ${paddle_attention_pack_index} paddle_attention_pack_option)
            list(APPEND paddle_attention_pack_options
                "GGML_SPACEMIT_OCR_FA_${paddle_attention_pack_option}=${paddle_attention_pack_value}")
        endforeach()
        llama_test(test-paddle-attention-pack NAME "${name}" LABEL spacemit
            ARGS ${paddle_attention_pack_action} "${paddle_attention_pack_reference}")
        set_tests_properties("${name}" PROPERTIES
            ENVIRONMENT "${paddle_attention_pack_options}" RUN_SERIAL TRUE TIMEOUT 1200)
        if (write_reference)
            set_tests_properties("${name}" PROPERTIES FIXTURES_SETUP paddle_attention_pack_reference)
        else()
            set_tests_properties("${name}" PROPERTIES FIXTURES_REQUIRED paddle_attention_pack_reference)
        endif()
    endfunction()

    paddle_attention_pack_add(test-paddle-attention-pack-reference 000000 TRUE)
    # Re-execute the exact reference mechanism to distinguish fixture/generator
    # drift from a candidate mismatch. This also verifies reference stability.
    paddle_attention_pack_add(test-paddle-attention-pack-control 000000 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full 100000 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-lean-alone 010000 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-query-alone 001000 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-half-alone 000100 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-softmax-alone 000010 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-compact-alone 000001 FALSE)
    # LEAN/QUERY are inactive with FULL_PACK=0. These combinations exercise
    # their actual branches separately, before the cumulative rollout.
    paddle_attention_pack_add(test-paddle-attention-pack-full-lean 110000 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full-query 101000 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full-half 100100 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full-softmax 100010 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full-compact 100001 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full-lean-query 111000 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full-lean-query-half 111100 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-full-lean-query-half-softmax 111110 FALSE)
    paddle_attention_pack_add(test-paddle-attention-pack-all 111111 FALSE)
endif()

# K4304 K1440 staging: exact comparisons include activation refresh and real M848.
if (GGML_SPACEMIT)
    llama_build(test-paddle-kblock-plan.cpp LINK ggml)
    llama_test(test-paddle-kblock-plan LABEL spacemit)
    set_tests_properties(test-paddle-kblock-plan PROPERTIES RUN_SERIAL TRUE)
    llama_test(test-paddle-f16-ime2 NAME test-paddle-f16-ime2-kblock-1440 LABEL spacemit
        ARGS --compare-reference "${paddle_f16_reference}")
    set_tests_properties(test-paddle-f16-ime2-kblock-1440 PROPERTIES
        ENVIRONMENT "${paddle_f16_env};GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=4;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=64;GGML_SPACEMIT_PADDLE_F16_M16=1;GGML_SPACEMIT_PADDLE_F16_KBLOCK=1440"
        FIXTURES_REQUIRED paddle_f16_panel_reference RUN_SERIAL TRUE)
    llama_test(test-paddle-f16-bias NAME test-paddle-f16-bias-kblock1440 LABEL spacemit
        ARGS --compare-reference "${paddle_f16_bias_reference}")
    set_tests_properties(test-paddle-f16-bias-kblock1440 PROPERTIES
        ENVIRONMENT "${paddle_f16_bias_env};GGML_SPACEMIT_PADDLE_F16_BIAS_FUSE=1;GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=4;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=64;GGML_SPACEMIT_PADDLE_F16_M16=1;GGML_SPACEMIT_PADDLE_F16_KBLOCK=1440"
        FIXTURES_REQUIRED paddle_f16_bias_reference RUN_SERIAL TRUE)
endif()
# Strict independent three-node F16 GEMM+bias+GELU byte regression.
if (GGML_SPACEMIT)
    llama_build(test-paddle-f16-bias-gelu.cpp LINK ggml)
    set(paddle_f16_bias_gelu_reference "${CMAKE_CURRENT_BINARY_DIR}/paddle-f16-bias-gelu-reference.bin")
    set(paddle_f16_bias_gelu_env "GGML_SPACEMIT_BACKEND=1;GGML_SPACEMIT_WORKERS=6;SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13;GGML_SPACEMIT_SHARED_MODE=PINNED_TCM;GGML_SPACEMIT_STRICT_AI_CORES=1;SPERT_SHARED_MODE=TCM;SPERT_BACKEND=spacemit-k3;SPERT_RT_PRIO=0")
    llama_test(test-paddle-f16-bias-gelu NAME test-paddle-f16-bias-gelu-reference LABEL spacemit
        ARGS --write-reference "${paddle_f16_bias_gelu_reference}")
    set_tests_properties(test-paddle-f16-bias-gelu-reference PROPERTIES
        ENVIRONMENT "${paddle_f16_bias_gelu_env};GGML_SPACEMIT_PADDLE_F16_GELU_FUSE=0;GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=1;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=8;GGML_SPACEMIT_PADDLE_F16_M16=0;GGML_SPACEMIT_PADDLE_F16_KBLOCK=0"
        FIXTURES_SETUP paddle_f16_bias_gelu_reference RUN_SERIAL TRUE)
    foreach(paddle_f16_bias_gelu_mode IN ITEMS base m16 kblock1024 kblock1440)
        set(paddle_f16_bias_gelu_options "GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=1;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=8;GGML_SPACEMIT_PADDLE_F16_M16=0;GGML_SPACEMIT_PADDLE_F16_KBLOCK=0")
        if (NOT paddle_f16_bias_gelu_mode STREQUAL "base")
            set(paddle_f16_bias_gelu_options "GGML_SPACEMIT_PADDLE_F16_PANEL_TILES=4;GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH=64;GGML_SPACEMIT_PADDLE_F16_M16=1;GGML_SPACEMIT_PADDLE_F16_KBLOCK=0")
        endif()
        if (paddle_f16_bias_gelu_mode STREQUAL "kblock1024")
            string(REPLACE "KBLOCK=0" "KBLOCK=1024" paddle_f16_bias_gelu_options "${paddle_f16_bias_gelu_options}")
        endif()
        if (paddle_f16_bias_gelu_mode STREQUAL "kblock1440")
            string(REPLACE "KBLOCK=0" "KBLOCK=1440" paddle_f16_bias_gelu_options "${paddle_f16_bias_gelu_options}")
        endif()
        set(paddle_f16_bias_gelu_test "test-paddle-f16-bias-gelu-${paddle_f16_bias_gelu_mode}")
        llama_test(test-paddle-f16-bias-gelu NAME "${paddle_f16_bias_gelu_test}" LABEL spacemit
            ARGS --compare-reference "${paddle_f16_bias_gelu_reference}")
        set_tests_properties("${paddle_f16_bias_gelu_test}" PROPERTIES
            ENVIRONMENT "${paddle_f16_bias_gelu_env};GGML_SPACEMIT_PADDLE_F16_GELU_FUSE=1;${paddle_f16_bias_gelu_options}"
            FIXTURES_REQUIRED paddle_f16_bias_gelu_reference RUN_SERIAL TRUE)
    endforeach()
endif()
# Preserve all output bits while changing query and packed-output layouts.
if (GGML_SPACEMIT)
    llama_build(test-paddle-attention-layout.cpp LINK ggml)
    set(paddle_attention_layout_reference "${CMAKE_CURRENT_BINARY_DIR}/paddle-attention-layout-reference.bin")
    set(paddle_attention_layout_base "GGML_SPACEMIT_OCR_FA_FULL_PACK=1;GGML_SPACEMIT_OCR_FA_LEAN_PREPARE=1;GGML_SPACEMIT_OCR_FA_FUSED_QUERY=1;GGML_SPACEMIT_OCR_FA_HALF_PACK=1;GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK=1;GGML_SPACEMIT_OCR_FA_COMPACT=1")
    set(paddle_attention_layout_zero "GGML_SPACEMIT_OCR_FA_FULL_PACK=0;GGML_SPACEMIT_OCR_FA_LEAN_PREPARE=0;GGML_SPACEMIT_OCR_FA_FUSED_QUERY=0;GGML_SPACEMIT_OCR_FA_HALF_PACK=0;GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK=0;GGML_SPACEMIT_OCR_FA_COMPACT=0")
    set(paddle_attention_layout_extra_zero "GGML_SPACEMIT_OCR_FA_Q32=0;GGML_SPACEMIT_OCR_FA_FIXED_INDEX=0;GGML_SPACEMIT_OCR_FA_P_STORE64=0;GGML_SPACEMIT_OCR_FA_QK_SCATTER=0")
    llama_test(test-paddle-attention-layout NAME test-paddle-attention-layout-reference LABEL spacemit ARGS --write-reference "${paddle_attention_layout_reference}")
    set_tests_properties(test-paddle-attention-layout-reference PROPERTIES ENVIRONMENT "${paddle_attention_layout_zero};${paddle_attention_layout_extra_zero}" FIXTURES_SETUP paddle_attention_layout_reference RUN_SERIAL TRUE)
    foreach(paddle_attention_layout_mode IN ITEMS base q32 index store64 scatter combined)
        set(paddle_attention_layout_env "${paddle_attention_layout_base};${paddle_attention_layout_extra_zero}")
        if (paddle_attention_layout_mode STREQUAL "q32" OR paddle_attention_layout_mode STREQUAL "combined")
            string(REPLACE "FA_Q32=0" "FA_Q32=1" paddle_attention_layout_env "${paddle_attention_layout_env}")
        endif()
        if (paddle_attention_layout_mode STREQUAL "index" OR paddle_attention_layout_mode STREQUAL "combined")
            string(REPLACE "FA_FIXED_INDEX=0" "FA_FIXED_INDEX=1" paddle_attention_layout_env "${paddle_attention_layout_env}")
        endif()
        if (paddle_attention_layout_mode STREQUAL "store64" OR paddle_attention_layout_mode STREQUAL "combined")
            string(REPLACE "FA_P_STORE64=0" "FA_P_STORE64=1" paddle_attention_layout_env "${paddle_attention_layout_env}")
        endif()
        if (paddle_attention_layout_mode STREQUAL "scatter" OR paddle_attention_layout_mode STREQUAL "combined")
            string(REPLACE "FA_QK_SCATTER=0" "FA_QK_SCATTER=1" paddle_attention_layout_env "${paddle_attention_layout_env}")
        endif()
        set(paddle_attention_layout_test "test-paddle-attention-layout-${paddle_attention_layout_mode}")
        llama_test(test-paddle-attention-layout NAME "${paddle_attention_layout_test}" LABEL spacemit ARGS --compare-reference "${paddle_attention_layout_reference}")
        set_tests_properties("${paddle_attention_layout_test}" PROPERTIES ENVIRONMENT "${paddle_attention_layout_env}" FIXTURES_REQUIRED paddle_attention_layout_reference RUN_SERIAL TRUE)
    endforeach()
endif()
