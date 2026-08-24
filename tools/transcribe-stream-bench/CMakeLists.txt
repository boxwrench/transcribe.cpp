# transcribe-stream-bench: cache-aware streaming latency measurement.

add_executable(transcribe-stream-bench
    main.cpp
)

target_link_libraries(transcribe-stream-bench
    PRIVATE
        transcribe
        transcribe-common-example
)

transcribe_apply_warnings(transcribe-stream-bench)

set_target_properties(transcribe-stream-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
