# EP and GGML link the same runtime; every model call shares the AI gate.
set(OCR_EP_DIR "" CACHE PATH "SpacemiT EP build or install prefix")
set(OCR_ORT_DIR "" CACHE PATH "Matching SpacemiT ONNX Runtime install prefix")
find_path(OCR_ORT_INCLUDE onnxruntime_cxx_api.h PATHS "${OCR_ORT_DIR}"
          PATH_SUFFIXES include/onnxruntime include/onnxruntime/core/session include
          NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
find_library(OCR_ORT_LIB onnxruntime PATHS "${OCR_ORT_DIR}/lib"
             NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
find_library(OCR_EP_LIB spacemit_ep PATHS "${OCR_EP_DIR}/lib" "${OCR_EP_DIR}/src"
             NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
if(NOT TARGET ggml-spacemit)
    message(FATAL_ERROR "OCR_SPACEMIT_EP requires GGML_SPACEMIT")
endif()
add_library(ocr-ep SHARED ocr-ep.cpp ocr-ep.h)
target_compile_features(ocr-ep PRIVATE cxx_std_17)
target_include_directories(ocr-ep PUBLIC . PRIVATE "${OCR_ORT_INCLUDE}" "${PROJECT_SOURCE_DIR}/ggml/include")
target_link_libraries(ocr-ep PUBLIC ggml-spacemit ggml-cpu ggml-base "${OCR_EP_LIB}" "${OCR_ORT_LIB}")
target_link_options(ocr-ep PRIVATE "-Wl,-z,defs")
install(TARGETS ocr-ep LIBRARY)
option(OCR_EP_BUILD_PROBE "Build the EP numerical verification client" OFF)
if(OCR_EP_BUILD_PROBE)
    add_executable(ocr-ep-probe ocr-ep-probe.cpp)
    target_link_libraries(ocr-ep-probe PRIVATE ocr-ep)
endif()
