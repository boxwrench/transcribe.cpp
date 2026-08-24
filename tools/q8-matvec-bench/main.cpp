#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct options {
    int k = 1024;
    int m = 1024;
    int n = 1;
    int warmup = 20;
    int iters = 500;
    std::string json_out;
};

struct graph_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * input = nullptr;
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

    graph_fixture(const graph_fixture &) = delete;
    graph_fixture & operator=(const graph_fixture &) = delete;
    graph_fixture() = default;
};

struct timing_summary {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

bool parse_positive(const char * text, int & value) {
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s [--k N] [--m N] [--n N] [--warmup N] [--iters N] --json-out PATH\n",
            argv0);
}

bool parse_options(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            return false;
        }
        const char * value = argv[++i];
        if (arg == "--k") {
            if (!parse_positive(value, opts.k)) {
                return false;
            }
        } else if (arg == "--m") {
            if (!parse_positive(value, opts.m)) {
                return false;
            }
        } else if (arg == "--n") {
            if (!parse_positive(value, opts.n)) {
                return false;
            }
        } else if (arg == "--warmup") {
            if (!parse_positive(value, opts.warmup)) {
                return false;
            }
        } else if (arg == "--iters") {
            if (!parse_positive(value, opts.iters)) {
                return false;
            }
        } else if (arg == "--json-out") {
            opts.json_out = value;
        } else {
            return false;
        }
    }
    return !opts.json_out.empty() && opts.k % ggml_blck_size(GGML_TYPE_Q8_0) == 0;
}

bool init_graph(graph_fixture & fixture, ggml_backend_t backend, const options & opts,
        const std::vector<uint8_t> & quantized, const std::vector<float> & input) {
    const size_t arena_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false);
    ggml_init_params params = {
        /* .mem_size   = */ arena_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    fixture.ctx = ggml_init(params);
    if (fixture.ctx == nullptr) {
        return false;
    }

    fixture.weights = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_Q8_0, opts.k, opts.m);
    fixture.input = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F32, opts.k, opts.n);
    fixture.output = ggml_mul_mat(fixture.ctx, fixture.weights, fixture.input);
    fixture.graph = ggml_new_graph_custom(fixture.ctx, 8, false);
    ggml_build_forward_expand(fixture.graph, fixture.output);

    if (!ggml_backend_supports_op(backend, fixture.output)) {
        return false;
    }
    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, backend);
    if (fixture.buffer == nullptr) {
        return false;
    }
    if (quantized.size() != ggml_nbytes(fixture.weights) ||
        input.size() != static_cast<size_t>(ggml_nelements(fixture.input))) {
        return false;
    }
    ggml_backend_tensor_set(fixture.weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(fixture.input, input.data(), 0, input.size() * sizeof(float));
    return true;
}

bool compute(ggml_backend_t backend, graph_fixture & fixture, std::vector<float> * output) {
    if (ggml_backend_graph_compute(backend, fixture.graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (output != nullptr) {
        output->resize(ggml_nelements(fixture.output));
        ggml_backend_tensor_get(fixture.output, output->data(), 0, output->size() * sizeof(float));
    }
    return true;
}

double percentile(const std::vector<double> & sorted, double quantile) {
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

timing_summary summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    double total = 0.0;
    for (double value : samples) {
        total += value;
    }
    return {
        total / static_cast<double>(samples.size()),
        percentile(samples, 0.50),
        percentile(samples, 0.95),
        percentile(samples, 0.99),
        samples.front(),
        samples.back(),
    };
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        usage(argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_t gpu_backend = ggml_backend_init_best();
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu_backend == nullptr || cpu_backend == nullptr ||
            ggml_backend_dev_type(ggml_backend_get_device(gpu_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        std::fprintf(stderr, "error: GPU and CPU backends are required\n");
        if (gpu_backend != nullptr) {
            ggml_backend_free(gpu_backend);
        }
        if (cpu_backend != nullptr) {
            ggml_backend_free(cpu_backend);
        }
        return 1;
    }

    std::vector<float> weights(static_cast<size_t>(opts.k) * opts.m);
    std::vector<float> input(static_cast<size_t>(opts.k) * opts.n);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = 0.125f * std::sin(static_cast<float>(i % 4093) * 0.017f);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.25f * std::cos(static_cast<float>(i % 1021) * 0.031f);
    }

    std::vector<uint8_t> quantized(ggml_row_size(GGML_TYPE_Q8_0, opts.k) * static_cast<size_t>(opts.m));
    const size_t quantized_size = ggml_quantize_chunk(
            GGML_TYPE_Q8_0, weights.data(), quantized.data(), 0, opts.m, opts.k, nullptr);
    if (quantized_size != quantized.size()) {
        std::fprintf(stderr, "error: Q8_0 quantization size mismatch\n");
        ggml_backend_free(gpu_backend);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    int result = 0;
    {
        graph_fixture gpu_graph;
        graph_fixture cpu_graph;
        if (!init_graph(gpu_graph, gpu_backend, opts, quantized, input) ||
                !init_graph(cpu_graph, cpu_backend, opts, quantized, input)) {
            std::fprintf(stderr, "error: failed to initialize benchmark graphs\n");
            result = 1;
        } else {
            std::vector<float> gpu_output;
            std::vector<float> cpu_output;
            if (!compute(cpu_backend, cpu_graph, &cpu_output) || !compute(gpu_backend, gpu_graph, &gpu_output)) {
                std::fprintf(stderr, "error: initial graph computation failed\n");
                result = 1;
            } else {
                double max_abs_error = 0.0;
                double max_rel_error = 0.0;
                for (size_t i = 0; i < gpu_output.size(); ++i) {
                    const double abs_error = std::abs(static_cast<double>(gpu_output[i]) - cpu_output[i]);
                    const double denominator = std::max(std::abs(static_cast<double>(cpu_output[i])), 1.0e-6);
                    max_abs_error = std::max(max_abs_error, abs_error);
                    max_rel_error = std::max(max_rel_error, abs_error / denominator);
                }

                for (int i = 0; i < opts.warmup && result == 0; ++i) {
                    if (!compute(gpu_backend, gpu_graph, nullptr)) {
                        result = 1;
                    }
                }

                std::vector<double> samples;
                samples.reserve(opts.iters);
                for (int i = 0; i < opts.iters && result == 0; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    if (!compute(gpu_backend, gpu_graph, nullptr)) {
                        result = 1;
                        break;
                    }
                    const auto stop = std::chrono::steady_clock::now();
                    samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
                }

                if (result == 0) {
                    const timing_summary timing = summarize(samples);
                    std::ofstream out(opts.json_out);
                    if (!out) {
                        std::fprintf(stderr, "error: cannot open JSON output\n");
                        result = 1;
                    } else {
                        out << std::fixed << std::setprecision(6)
                            << "{\n"
                            << "  \"schema\": \"q8-matvec-bench-v1\",\n"
                            << "  \"backend\": \"" << ggml_backend_name(gpu_backend) << "\",\n"
                            << "  \"shape\": {\"k\": " << opts.k << ", \"m\": " << opts.m
                            << ", \"n\": " << opts.n << "},\n"
                            << "  \"warmup\": " << opts.warmup << ",\n"
                            << "  \"iterations\": " << opts.iters << ",\n"
                            << "  \"timing_us\": {\"mean\": " << timing.mean_us
                            << ", \"p50\": " << timing.p50_us << ", \"p95\": " << timing.p95_us
                            << ", \"p99\": " << timing.p99_us << ", \"min\": " << timing.min_us
                            << ", \"max\": " << timing.max_us << "},\n"
                            << "  \"correctness\": {\"max_abs_error\": " << max_abs_error
                            << ", \"max_rel_error\": " << max_rel_error << "}\n"
                            << "}\n";
                    }
                }
            }
        }
    }

    ggml_backend_free(gpu_backend);
    ggml_backend_free(cpu_backend);
    return result;
}
