add_executable(norm-fusion-check main.cpp)
target_link_libraries(norm-fusion-check PRIVATE ggml)
transcribe_apply_warnings(norm-fusion-check)
set_target_properties(norm-fusion-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
