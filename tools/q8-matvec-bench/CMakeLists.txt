# q8-matvec-bench: shape-specific ggml Q8_0 backend benchmark.

add_executable(q8-matvec-bench
    main.cpp
)

target_link_libraries(q8-matvec-bench
    PRIVATE
        ggml
)

transcribe_apply_warnings(q8-matvec-bench)

set_target_properties(q8-matvec-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
