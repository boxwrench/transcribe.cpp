// transcribe-stream-bench - per-call streaming latency measurement.

#include "transcribe.h"
#include "transcribe/parakeet.h"
#include "wav.h"

#ifdef TRANSCRIBE_ROCTX_PROFILE
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct bench_args {
    std::string                model_path;
    std::string                sample_path;
    std::string                json_out;
    std::string                language;
    int                        iters        = 10;
    int                        warmup       = 2;
    int                        feed_ms      = 80;
    int                        att_right    = -1;
    int                        n_threads    = 0;
    int                        device_index = -1;
    bool                       quiet        = false;
    transcribe_backend_request backend      = TRANSCRIBE_BACKEND_AUTO;
};

struct call_timing {
    int64_t input_received_ms = 0;
    double  wall_ms           = 0.0;
    bool    result_changed    = false;
};

struct iter_result {
    std::vector<call_timing> feeds;
    double                   finalize_ms             = 0.0;
    double                   request_wall_ms         = 0.0;
    double                   first_result_compute_ms = -1.0;
    double                   mel_ms                  = 0.0;
    double                   encode_ms               = 0.0;
    double                   decode_ms               = 0.0;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s --model PATH --sample PATH [options]\n"
                 "options:\n"
                 "  --model PATH       GGUF model file (required)\n"
                 "  --sample PATH      16 kHz WAV file (required)\n"
                 "  --iters N          measured streams (default 10)\n"
                 "  --warmup N         untimed streams (default 2)\n"
                 "  --feed-ms N        PCM feed cadence in ms (default 80)\n"
                 "  --att-right N      Parakeet right-context value\n"
                 "  --language TAG     language hint (for example en-US)\n"
                 "  --backend KIND     auto|cpu|cpu_accel|vulkan|cuda|rocm\n"
                 "  --device N         exact index from transcribe-cli --list-devices\n"
                 "  --threads N        CPU threads (default 0 = library default)\n"
                 "  --json-out PATH    write JSON to PATH (default stdout)\n"
                 "  --quiet            suppress progress output\n",
                 argv0);
}

bool parse_backend(const char * value, transcribe_backend_request & out) {
    if (std::strcmp(value, "auto") == 0) {
        out = TRANSCRIBE_BACKEND_AUTO;
    } else if (std::strcmp(value, "cpu") == 0) {
        out = TRANSCRIBE_BACKEND_CPU;
    } else if (std::strcmp(value, "cpu_accel") == 0) {
        out = TRANSCRIBE_BACKEND_CPU_ACCEL;
    } else if (std::strcmp(value, "vulkan") == 0) {
        out = TRANSCRIBE_BACKEND_VULKAN;
    } else if (std::strcmp(value, "cuda") == 0) {
        out = TRANSCRIBE_BACKEND_CUDA;
    } else if (std::strcmp(value, "rocm") == 0) {
        out = TRANSCRIBE_BACKEND_ROCM;
    } else {
        return false;
    }
    return true;
}

bool parse_args(int argc, char ** argv, bench_args & out) {
    auto value = [&](int & i, const char * name) -> const char * {
        if (++i >= argc) {
            std::fprintf(stderr, "error: %s requires a value\n", name);
            return nullptr;
        }
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--quiet") {
            out.quiet = true;
        } else if (arg == "--model") {
            const char * v = value(i, "--model");
            if (!v) {
                return false;
            }
            out.model_path = v;
        } else if (arg == "--sample") {
            const char * v = value(i, "--sample");
            if (!v) {
                return false;
            }
            out.sample_path = v;
        } else if (arg == "--json-out") {
            const char * v = value(i, "--json-out");
            if (!v) {
                return false;
            }
            out.json_out = v;
        } else if (arg == "--language") {
            const char * v = value(i, "--language");
            if (!v) {
                return false;
            }
            out.language = v;
        } else if (arg == "--iters") {
            const char * v = value(i, "--iters");
            if (!v) {
                return false;
            }
            out.iters = std::atoi(v);
        } else if (arg == "--warmup") {
            const char * v = value(i, "--warmup");
            if (!v) {
                return false;
            }
            out.warmup = std::atoi(v);
        } else if (arg == "--feed-ms") {
            const char * v = value(i, "--feed-ms");
            if (!v) {
                return false;
            }
            out.feed_ms = std::atoi(v);
        } else if (arg == "--att-right") {
            const char * v = value(i, "--att-right");
            if (!v) {
                return false;
            }
            out.att_right = std::atoi(v);
        } else if (arg == "--threads") {
            const char * v = value(i, "--threads");
            if (!v) {
                return false;
            }
            out.n_threads = std::atoi(v);
        } else if (arg == "--device") {
            const char * v = value(i, "--device");
            if (!v) {
                return false;
            }
            out.device_index = std::atoi(v);
        } else if (arg == "--backend") {
            const char * v = value(i, "--backend");
            if (!v || !parse_backend(v, out.backend)) {
                std::fprintf(stderr, "error: invalid --backend\n");
                return false;
            }
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", arg.c_str());
            return false;
        }
    }
    if (out.model_path.empty() || out.sample_path.empty() || out.iters < 1 || out.warmup < 0 || out.feed_ms < 1 ||
        out.att_right < -1 || out.n_threads < 0 || out.device_index < -1) {
        return false;
    }
    return true;
}

std::string json_escape(const char * value) {
    std::string out;
    for (const char * p = value ? value : ""; *p; ++p) {
        if (*p == '"') {
            out += "\\\"";
        } else if (*p == '\\') {
            out += "\\\\";
        } else if (*p == '\n') {
            out += "\\n";
        } else if (*p == '\r') {
            out += "\\r";
        } else if (*p == '\t') {
            out += "\\t";
        } else {
            out += *p;
        }
    }
    return out;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void append_fmt(std::string & out, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    char      local[512];
    const int n = std::vsnprintf(local, sizeof(local), fmt, ap);
    va_end(ap);
    if (n >= 0 && static_cast<size_t>(n) < sizeof(local)) {
        out.append(local, static_cast<size_t>(n));
    } else if (n >= 0) {
        std::string large(static_cast<size_t>(n) + 1, '\0');
        std::vsnprintf(large.data(), large.size(), fmt, copy);
        out.append(large.data(), static_cast<size_t>(n));
    }
    va_end(copy);
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo  = static_cast<size_t>(std::floor(pos));
    const size_t hi  = static_cast<size_t>(std::ceil(pos));
    const double mix = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - mix) + values[hi] * mix;
}

double mean(const std::vector<double> & values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

transcribe_status run_stream(transcribe_session *       ctx,
                             const bench_args &         args,
                             const std::vector<float> & pcm,
                             bool                       measure,
                             iter_result &              result) {
    transcribe_stream_reset(ctx);
    transcribe_reset_timings(ctx);

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.language = args.language.empty() ? nullptr : args.language.c_str();

    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    transcribe_parakeet_stream_ext parakeet;
    transcribe_parakeet_stream_ext_init(&parakeet);
    if (args.att_right >= 0 && transcribe_model_accepts_ext_kind(transcribe_get_model(ctx), TRANSCRIBE_EXT_SLOT_STREAM,
                                                                 TRANSCRIBE_EXT_KIND_PARAKEET_STREAM)) {
        parakeet.att_context_right = args.att_right;
        sp.family                  = &parakeet.ext;
    }

    transcribe_status st = transcribe_stream_begin(ctx, &rp, &sp);
    if (st != TRANSCRIBE_OK) {
        return st;
    }

    const auto   request_begin = std::chrono::steady_clock::now();
    const size_t chunk_samples = static_cast<size_t>(args.feed_ms) * 16000 / 1000;
    size_t       pos           = 0;
    double       cumulative_ms = 0.0;
    while (pos < pcm.size()) {
        const size_t             take = std::min(chunk_samples, pcm.size() - pos);
        transcribe_stream_update update;
        transcribe_stream_update_init(&update);
        const auto begin = std::chrono::steady_clock::now();
        st               = transcribe_stream_feed(ctx, pcm.data() + pos, static_cast<int>(take), &update);
        const auto end   = std::chrono::steady_clock::now();
        if (st != TRANSCRIBE_OK) {
            return st;
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        cumulative_ms += wall_ms;
        if (measure) {
            result.feeds.push_back({ update.input_received_ms, wall_ms, update.result_changed });
            const char * text = transcribe_full_text(ctx);
            if (result.first_result_compute_ms < 0.0 && update.result_changed && text != nullptr && text[0] != '\0') {
                result.first_result_compute_ms = cumulative_ms;
            }
        }
        pos += take;
    }

    transcribe_stream_update final_update;
    transcribe_stream_update_init(&final_update);
    const auto finalize_begin = std::chrono::steady_clock::now();
    st                        = transcribe_stream_finalize(ctx, &final_update);
    const auto finalize_end   = std::chrono::steady_clock::now();
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    if (measure) {
        result.finalize_ms     = std::chrono::duration<double, std::milli>(finalize_end - finalize_begin).count();
        result.request_wall_ms = std::chrono::duration<double, std::milli>(finalize_end - request_begin).count();
        if (result.first_result_compute_ms < 0.0) {
            result.first_result_compute_ms = result.request_wall_ms;
        }
        transcribe_timings timings;
        transcribe_timings_init(&timings);
        (void) transcribe_get_timings(ctx, &timings);
        result.mel_ms    = timings.mel_ms;
        result.encode_ms = timings.encode_ms;
        result.decode_ms = timings.decode_ms;
    }
    return TRANSCRIBE_OK;
}

}  // namespace

int main(int argc, char ** argv) {
    bench_args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.quiet) {
        transcribe_log_set(nullptr, nullptr);
    }

    std::vector<float> pcm;
    std::string        wav_error;
    if (!transcribe_cli::load_wav_mono_16k(args.sample_path, pcm, wav_error)) {
        std::fprintf(stderr, "error: wav: %s\n", wav_error.c_str());
        return EXIT_FAILURE;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = args.backend;
    mp.device  = args.device_index >= 0 ? transcribe_device_get(args.device_index) : nullptr;
    if (args.device_index >= 0 && mp.device == nullptr) {
        std::fprintf(stderr, "error: device %d is unavailable\n", args.device_index);
        return EXIT_FAILURE;
    }
    transcribe_model * model = nullptr;
    transcribe_status  st    = transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "error: model load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }
    transcribe_session_params cp;
    transcribe_session_params_init(&cp);
    cp.n_threads             = args.n_threads;
    transcribe_session * ctx = nullptr;
    st                       = transcribe_session_init(model, &cp, &ctx);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "error: session init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < args.warmup; ++i) {
        iter_result ignored;
        st = run_stream(ctx, args, pcm, false, ignored);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "error: warmup: %s\n", transcribe_status_string(st));
            transcribe_session_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    std::vector<iter_result> results;
    std::vector<double>      feed_values;
    std::vector<double>      request_values;
    std::vector<double>      first_values;
    std::vector<double>      finalize_values;
    results.reserve(static_cast<size_t>(args.iters));
    for (int i = 0; i < args.iters; ++i) {
        if (!args.quiet) {
            std::fprintf(stderr, "stream %d/%d\n", i + 1, args.iters);
        }
        iter_result result;
#ifdef TRANSCRIBE_ROCTX_PROFILE
        char range_name[64];
        std::snprintf(range_name, sizeof(range_name), "transcribe|request|iter=%d", i);
        roctxRangePushA(range_name);
#endif
        st = run_stream(ctx, args, pcm, true, result);
#ifdef TRANSCRIBE_ROCTX_PROFILE
        roctxRangePop();
#endif
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "error: stream %d: %s\n", i + 1, transcribe_status_string(st));
            transcribe_session_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
        for (const call_timing & call : result.feeds) {
            feed_values.push_back(call.wall_ms);
        }
        request_values.push_back(result.request_wall_ms);
        first_values.push_back(result.first_result_compute_ms);
        finalize_values.push_back(result.finalize_ms);
        results.push_back(std::move(result));
    }

    const char * backend = transcribe_model_backend(model);
    const char * text    = transcribe_full_text(ctx);
    std::string  token_ids;
    for (int i = 0; i < transcribe_n_tokens(ctx); ++i) {
        transcribe_token token;
        transcribe_token_init(&token);
        (void) transcribe_get_token(ctx, i, &token);
        if (i > 0) {
            token_ids += ',';
        }
        token_ids += std::to_string(token.id);
    }

    std::string out;
    out += "{\n  \"schema\": \"transcribe-stream-bench-v1\",\n";
    append_fmt(out, "  \"model_path\": \"%s\",\n", json_escape(args.model_path.c_str()).c_str());
    append_fmt(out, "  \"sample_path\": \"%s\",\n", json_escape(args.sample_path.c_str()).c_str());
    append_fmt(out, "  \"sample_duration_s\": %.3f,\n", static_cast<double>(pcm.size()) / 16000.0);
    append_fmt(out, "  \"backend\": \"%s\",\n", json_escape(backend).c_str());
    append_fmt(out, "  \"device_index\": %d,\n", args.device_index);
    append_fmt(out, "  \"language\": \"%s\",\n", json_escape(args.language.c_str()).c_str());
    append_fmt(out, "  \"feed_ms\": %d,\n  \"att_right\": %d,\n", args.feed_ms, args.att_right);
    append_fmt(out, "  \"warmup\": %d,\n  \"iters\": %d,\n", args.warmup, args.iters);
    out += "  \"per_iter\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const iter_result & row = results[i];
        append_fmt(out,
                   "    {\"request_wall_ms\":%.3f,\"first_result_compute_ms\":%.3f,"
                   "\"finalize_ms\":%.3f,\"mel_ms\":%.3f,\"encode_ms\":%.3f,\"decode_ms\":%.3f,\"feeds\":[",
                   row.request_wall_ms, row.first_result_compute_ms, row.finalize_ms, row.mel_ms, row.encode_ms,
                   row.decode_ms);
        for (size_t j = 0; j < row.feeds.size(); ++j) {
            const call_timing & call = row.feeds[j];
            append_fmt(out, "{\"input_received_ms\":%lld,\"wall_ms\":%.3f,\"result_changed\":%s}%s",
                       static_cast<long long>(call.input_received_ms), call.wall_ms,
                       call.result_changed ? "true" : "false", j + 1 == row.feeds.size() ? "" : ",");
        }
        append_fmt(out, "]}%s\n", i + 1 == results.size() ? "" : ",");
    }
    out += "  ],\n  \"summary\": {\n";
    append_fmt(out, "    \"chunk_ms\": {\"mean\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f,\"max\":%.3f},\n",
               mean(feed_values), percentile(feed_values, 0.50), percentile(feed_values, 0.95),
               percentile(feed_values, 0.99),
               feed_values.empty() ? 0.0 : *std::max_element(feed_values.begin(), feed_values.end()));
    append_fmt(out, "    \"request_wall_ms\": {\"mean\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},\n",
               mean(request_values), percentile(request_values, 0.50), percentile(request_values, 0.95),
               percentile(request_values, 0.99));
    append_fmt(out, "    \"first_result_compute_ms\": {\"mean\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f},\n",
               mean(first_values), percentile(first_values, 0.50), percentile(first_values, 0.95),
               percentile(first_values, 0.99));
    append_fmt(out, "    \"finalize_ms\": {\"mean\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f}\n",
               mean(finalize_values), percentile(finalize_values, 0.50), percentile(finalize_values, 0.95),
               percentile(finalize_values, 0.99));
    out += "  },\n";
    append_fmt(out, "  \"hyp_text\": \"%s\",\n", json_escape(text).c_str());
    append_fmt(out, "  \"token_ids_csv\": \"%s\"\n}\n", token_ids.c_str());

    bool write_ok = true;
    if (args.json_out.empty()) {
        std::fwrite(out.data(), 1, out.size(), stdout);
    } else {
        std::ofstream file(args.json_out, std::ios::binary | std::ios::trunc);
        file << out;
        write_ok = static_cast<bool>(file);
        if (!write_ok) {
            std::fprintf(stderr, "error: cannot write %s\n", args.json_out.c_str());
        }
    }

    transcribe_session_free(ctx);
    transcribe_model_free(model);
    return write_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
