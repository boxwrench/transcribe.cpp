#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct graph_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * gamma = nullptr;
    ggml_tensor * beta = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;

    ~graph_fixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

bool init_graph(graph_fixture & fixture, ggml_backend_t backend, int64_t nrows,
                const std::vector<float> & input,
                const std::vector<float> & gamma,
                const std::vector<float> & beta) {
    constexpr int64_t ncols = 1024;
    const size_t arena_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false);
    fixture.ctx = ggml_init({arena_size, nullptr, true});
    if (fixture.ctx == nullptr) {
        return false;
    }

    fixture.input = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F32, ncols, nrows);
    fixture.gamma = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_F32, ncols);
    fixture.beta = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_F32, ncols);
    ggml_tensor * norm = ggml_norm(fixture.ctx, fixture.input, 1.0e-5f);
    ggml_tensor * mul = ggml_mul(fixture.ctx, norm, fixture.gamma);
    fixture.output = ggml_add(fixture.ctx, mul, fixture.beta);
    fixture.graph = ggml_new_graph_custom(fixture.ctx, 8, false);
    ggml_build_forward_expand(fixture.graph, fixture.output);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, backend);
    if (fixture.buffer == nullptr || !ggml_backend_supports_op(backend, fixture.output)) {
        return false;
    }
    ggml_backend_tensor_set(fixture.input, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(fixture.gamma, gamma.data(), 0, gamma.size() * sizeof(float));
    ggml_backend_tensor_set(fixture.beta, beta.data(), 0, beta.size() * sizeof(float));
    return true;
}

bool compute(ggml_backend_t backend, graph_fixture & fixture, std::vector<float> & output) {
    if (ggml_backend_graph_compute(backend, fixture.graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    output.resize(ggml_nelements(fixture.output));
    ggml_backend_tensor_get(fixture.output, output.data(), 0, output.size() * sizeof(float));
    return true;
}

} // namespace

int main() {
    constexpr int64_t ncols = 1024;
    std::vector<float> gamma(ncols);
    std::vector<float> beta(ncols);
    for (size_t i = 0; i < gamma.size(); ++i) {
        gamma[i] = 0.8f + 0.2f * std::cos(static_cast<float>(i) * 0.019f);
        beta[i] = 0.05f * std::sin(static_cast<float>(i) * 0.029f);
    }

    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_best();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu == nullptr || cpu == nullptr ||
            ggml_backend_dev_type(ggml_backend_get_device(gpu)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        std::fprintf(stderr, "error: GPU and CPU backends are required\n");
        return 1;
    }

    int status = 0;
    std::printf("{\"backend\":\"%s\",\"shapes\":[", ggml_backend_name(gpu));
    for (int64_t nrows : {1, 2}) {
        std::vector<float> input(ncols * nrows);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = 0.75f * std::sin(static_cast<float>(i) * 0.013f) + 0.1f * std::cos(static_cast<float>(i) * 0.071f);
        }
        graph_fixture gpu_graph;
        graph_fixture cpu_graph;
        std::vector<float> gpu_output;
        std::vector<float> cpu_output;
        if (!init_graph(gpu_graph, gpu, nrows, input, gamma, beta) ||
                !init_graph(cpu_graph, cpu, nrows, input, gamma, beta) ||
                !compute(gpu, gpu_graph, gpu_output) || !compute(cpu, cpu_graph, cpu_output)) {
            std::fprintf(stderr, "error: graph setup or computation failed\n");
            status = 1;
        } else {
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            for (size_t i = 0; i < gpu_output.size(); ++i) {
                const double error = std::abs(static_cast<double>(gpu_output[i]) - cpu_output[i]);
                max_abs_error = std::max(max_abs_error, error);
                max_rel_error = std::max(max_rel_error,
                        error / std::max(std::abs(static_cast<double>(cpu_output[i])), 1.0e-6));
            }
            for (int i = 0; i < 20; ++i) {
                if (ggml_backend_graph_compute(gpu, gpu_graph.graph) != GGML_STATUS_SUCCESS) {
                    status = 1;
                }
            }
            if (status != 0) {
                std::fprintf(stderr, "error: graph warmup failed\n");
                break;
            }
            std::vector<double> samples;
            samples.reserve(1000);
            for (int i = 0; i < 1000 && status == 0; ++i) {
                const auto start = std::chrono::steady_clock::now();
                if (ggml_backend_graph_compute(gpu, gpu_graph.graph) != GGML_STATUS_SUCCESS) {
                    status = 1;
                    break;
                }
                const auto stop = std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
            }
            if (status != 0 || samples.empty()) {
                std::fprintf(stderr, "error: timed graph computation failed\n");
                status = 1;
                break;
            }
            std::sort(samples.begin(), samples.end());
            const double mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
            const double p50_us = samples[samples.size() / 2];
            const double p95_us = samples[static_cast<size_t>(0.95 * (samples.size() - 1))];
            std::printf("%s{\"ne\":[1024,%lld,1,1],\"max_abs_error\":%.9g,\"max_rel_error\":%.9g,"
                        "\"timing_us\":{\"mean\":%.6f,\"p50\":%.6f,\"p95\":%.6f}}",
                    nrows == 1 ? "" : ",", (long long) nrows, max_abs_error, max_rel_error,
                    mean_us, p50_us, p95_us);
            if (max_abs_error > 1.0e-4 || max_rel_error > 1.0e-4) {
                status = 1;
            }
        }
    }
    std::printf("]}\n");

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return status;
}
