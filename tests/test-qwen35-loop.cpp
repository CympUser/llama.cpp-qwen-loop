#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static double calc_nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = a[i];
        const double db = b[i];
        num += (da - db) * (da - db);
        den += da * da;
    }
    return den > 0.0 ? num / den : num;
}

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

struct init_params_t {
    size_t seed;
    float  stdev;
};

static void fill_synthetic_tensors(struct ggml_tensor * tensor, void * userdata) {
    const auto & p = *static_cast<const init_params_t *>(userdata);
    size_t seed = p.seed ^ std::hash<std::string>{}(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, p.stdev);

    const bool is_ssm_a = std::strstr(tensor->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; ++i) {
            const float val = dis(gen);
            tmp[i] = is_ssm_a ? -fabsf(val) : val;
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; ++i) {
            const float val = dis(gen);
            tmp[i] = ggml_fp32_to_fp16(is_ssm_a ? -fabsf(val) : val);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type in test-qwen35-loop");
    }
}

static gguf_context_ptr make_qwen35_fixture(
        uint32_t n_loops,
        const char * kv_mode,
        uint32_t n_vocab = 128,
        uint32_t n_embd = 128,
        uint32_t n_head = 2,
        uint32_t n_ff = 192) {
    const uint32_t n_layer = 4;
    const uint32_t head_dim = n_embd / n_head;
    const uint32_t n_ctx = 256;

    gguf_context_ptr ret(gguf_init_empty());
    gguf_context * g = ret.get();

    gguf_set_val_str(g, "general.architecture", "qwen35");
    gguf_set_val_u32(g, "qwen35.vocab_size", n_vocab);
    gguf_set_val_u32(g, "qwen35.context_length", n_ctx);
    gguf_set_val_u32(g, "qwen35.embedding_length", n_embd);
    gguf_set_val_u32(g, "qwen35.feed_forward_length", n_ff);
    gguf_set_val_u32(g, "qwen35.block_count", n_layer);
    gguf_set_val_u32(g, "qwen35.attention.head_count", n_head);
    gguf_set_val_u32(g, "qwen35.attention.head_count_kv", n_head);
    gguf_set_val_u32(g, "qwen35.attention.key_length", head_dim);
    gguf_set_val_u32(g, "qwen35.attention.value_length", head_dim);
    gguf_set_val_u32(g, "qwen35.rope.dimension_count", head_dim);
    gguf_set_val_f32(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_f32(g, "qwen35.rope.freq_base", 10000000.0f);

    const uint32_t rope_sections[4] = { 11, 11, 10, 0 };
    gguf_set_arr_data(g, "qwen35.rope.dimension_sections", GGUF_TYPE_UINT32, rope_sections, 4);

    // DeltaNet params
    gguf_set_val_u32(g, "qwen35.ssm.conv_kernel", 4);
    gguf_set_val_u32(g, "qwen35.ssm.inner_size", n_embd);
    gguf_set_val_u32(g, "qwen35.ssm.state_size", head_dim);
    gguf_set_val_u32(g, "qwen35.ssm.time_step_rank", n_head);
    gguf_set_val_u32(g, "qwen35.ssm.group_count", n_head);

    // 3 DeltaNet + 1 FullAttn, with FullAttn at block 2 inside the recurrent core [1, 2]
    const uint8_t recr_layers[4] = { 1, 1, 0, 1 };
    gguf_set_arr_data(g, "qwen35.attention.recurrent_layers", GGUF_TYPE_BOOL, recr_layers, 4);

    // Loop metadata
    gguf_set_val_u32(g, "qwen35.input_layer_count", 1);
    gguf_set_val_u32(g, "qwen35.recurrent_layer_count", 2);
    gguf_set_val_u32(g, "qwen35.output_layer_count", 1);
    gguf_set_val_u32(g, "qwen35.num_loops", n_loops);
    gguf_set_val_str(g, "qwen35.recurrent_kv_cache_mode", kv_mode);

    std::vector<std::string> tok_list(n_vocab);
    std::vector<const char *> tok_ptrs(n_vocab);
    std::vector<float> tok_scores(n_vocab, 0.0f);
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tok_list[i] = "t" + std::to_string(i);
        tok_ptrs[i] = tok_list[i].c_str();
    }
    gguf_set_val_str(g, "tokenizer.ggml.model", "test");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tok_ptrs.data(), tok_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, tok_scores.data(), tok_scores.size());

    return ret;
}

static std::string write_loop_adapter_gguf(
        const std::string & path,
        bool zero_gate,
        uint32_t max_loops = 6,
        uint32_t n_embd = 128,
        uint32_t n_ff = 192,
        uint32_t rank = 8) {
    gguf_context_ptr gctx(gguf_init_empty());
    gguf_set_val_str(gctx.get(), "general.type", "adapter");
    gguf_set_val_str(gctx.get(), "general.architecture", "qwen35");
    gguf_set_val_str(gctx.get(), "adapter.type", "lora");
    gguf_set_val_f32(gctx.get(), "adapter.lora.alpha", 16.0f);

    const size_t n_tensors_est = (max_loops - 1) * (3 + 6);
    ggml_init_params gp = {
        /*.mem_size   =*/ n_tensors_est * ggml_tensor_overhead() + 8ull * 1024ull * 1024ull,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context_ptr mctx(ggml_init(gp));
    GGML_ASSERT(mctx != nullptr);

    std::mt19937 gen(1337);
    std::normal_distribution<float> dis(0.0f, 0.08f);

    auto add_1d = [&](const std::string & name, float fill_val, float noise_std = 0.0f) {
        ggml_tensor * t = ggml_new_tensor_1d(mctx.get(), GGML_TYPE_F32, n_embd);
        ggml_set_name(t, name.c_str());
        float * d = reinterpret_cast<float *>(t->data);
        std::normal_distribution<float> nd(0.0f, noise_std > 0.0f ? noise_std : 1e-6f);
        for (uint32_t i = 0; i < n_embd; ++i) {
            d[i] = fill_val + (noise_std > 0.0f ? nd(gen) : 0.0f);
        }
        gguf_add_tensor(gctx.get(), t);
    };

    auto add_lora_pair = [&](const std::string & prefix, int64_t in_dim, int64_t out_dim, float scale_b) {
        ggml_tensor * ta = ggml_new_tensor_2d(mctx.get(), GGML_TYPE_F32, in_dim, rank);
        ggml_set_name(ta, (prefix + ".lora_a").c_str());
        float * da = reinterpret_cast<float *>(ta->data);
        for (int64_t i = 0; i < in_dim * (int64_t) rank; ++i) {
            da[i] = dis(gen);
        }
        gguf_add_tensor(gctx.get(), ta);

        ggml_tensor * tb = ggml_new_tensor_2d(mctx.get(), GGML_TYPE_F32, rank, out_dim);
        ggml_set_name(tb, (prefix + ".lora_b").c_str());
        float * db = reinterpret_cast<float *>(tb->data);
        for (int64_t i = 0; i < (int64_t) rank * out_dim; ++i) {
            db[i] = dis(gen) * scale_b;
        }
        gguf_add_tensor(gctx.get(), tb);
    };

    for (uint32_t k = 1; k < max_loops; ++k) {
        const float gate_val = zero_gate ? 0.0f : (0.35f + 0.08f * k);
        add_1d("loop." + std::to_string(k) + ".in_scale.weight", 0.5f, 0.05f);
        add_1d("loop." + std::to_string(k) + ".embd.weight", 0.03f * k, 0.02f);
        add_1d("loop." + std::to_string(k) + ".out_gate.weight", gate_val, zero_gate ? 0.0f : 0.05f);

        // DeltaNet layer (blk 1), FullAttn layer (blk 2), FFN down (blk 1)
        const float b_scale = 0.6f + 0.15f * k;
        add_lora_pair("loop." + std::to_string(k) + ".blk.1.ssm_out.weight", n_embd, n_embd, b_scale);
        add_lora_pair("loop." + std::to_string(k) + ".blk.2.attn_output.weight", n_embd, n_embd, b_scale);
        add_lora_pair("loop." + std::to_string(k) + ".blk.1.ffn_down.weight", n_ff, n_embd, b_scale);
    }

    if (!gguf_write_to_file(gctx.get(), path.c_str(), false)) {
        throw std::runtime_error("failed to write temp loop adapter GGUF: " + path);
    }
    return path;
}

static std::pair<llama_model_ptr, llama_context_ptr> load_fixture(
        gguf_context * gctx,
        ggml_backend_dev_t dev,
        size_t seed = 42,
        float stdev = 0.1f) {
    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    if (dev != nullptr) {
        mparams.devices = devs;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_ubatch = 64;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    init_params_t tp{seed, stdev};
    llama_model_ptr model(llama_model_init_from_user(gctx, fill_synthetic_tensors, &tp, mparams));
    if (!model) {
        throw std::runtime_error("llama_model_init_from_user failed");
    }
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        throw std::runtime_error("llama_init_from_model failed");
    }
    return { std::move(model), std::move(ctx) };
}

static std::vector<float> eval_logits_batch(
        llama_model * model,
        llama_context * ctx,
        const std::vector<llama_token> & tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.add(tokens[i], (llama_pos) i, 0, true);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        throw std::runtime_error("llama_process batch failed");
    }
    std::vector<float> out;
    out.reserve(tokens.size() * n_vocab);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * logits = llama_get_logits_ith(ctx, (int32_t) i);
        GGML_ASSERT(logits != nullptr);
        out.insert(out.end(), logits, logits + n_vocab);
    }
    return out;
}

static std::vector<float> eval_logits_step_by_step(
        llama_model * model,
        llama_context * ctx,
        const std::vector<llama_token> & tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_batch batch(ctx);
    std::vector<float> out;
    out.reserve(tokens.size() * n_vocab);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.clear();
        batch.add(tokens[i], (llama_pos) i, 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("llama_process step failed");
        }
        const float * logits = llama_get_logits_ith(ctx, 0);
        GGML_ASSERT(logits != nullptr);
        out.insert(out.end(), logits, logits + n_vocab);
    }
    return out;
}

int main() {
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);
    common_init();
    llama_backend_init();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    GGML_ASSERT(cpu_dev != nullptr);

    std::vector<ggml_backend_dev_t> test_devs = { cpu_dev };
    if (gpu_dev != nullptr) {
        test_devs.push_back(gpu_dev);
    }

    const std::vector<llama_token> tokens = { 3, 17, 42, 99, 5, 61, 88, 12, 73, 29, 104, 7, 55, 91, 22, 64 };

    const auto tmp_dir = std::filesystem::temp_directory_path();
    const std::string zero_path = (tmp_dir / "qwen35_loop_zero.gguf").string();
    const std::string spec_path = (tmp_dir / "qwen35_loop_spec.gguf").string();
    write_loop_adapter_gguf(zero_path, true);
    write_loop_adapter_gguf(spec_path, false);

    auto cleanup = [&]() {
        std::error_code ec;
        std::filesystem::remove(zero_path, ec);
        std::filesystem::remove(spec_path, ec);
    };

    try {
        // Test 1: Zero-init identity (K=6 + zero out_gate bridge == K=1 base model) on CPU & GPU
        printf("[Test 1] Zero-init identity check (K=6 vs K=1)...\n");
        for (ggml_backend_dev_t dev : test_devs) {
            auto g1 = make_qwen35_fixture(1, "per_loop");
            auto g6 = make_qwen35_fixture(6, "per_loop");
            auto [m1, c1] = load_fixture(g1.get(), dev);
            auto [m6, c6] = load_fixture(g6.get(), dev);
            GGML_ASSERT(llama_model_n_layer_physical(m6.get()) == 4);

            llama_adapter_lora * adapter = llama_adapter_lora_init(m6.get(), zero_path.c_str());
            GGML_ASSERT(adapter != nullptr);
            float scale = 1.0f;
            GGML_ASSERT(llama_set_adapters_lora(c6.get(), &adapter, 1, &scale) == 0);

            auto l1 = eval_logits_batch(m1.get(), c1.get(), tokens);
            auto l6 = eval_logits_batch(m6.get(), c6.get(), tokens);
            GGML_ASSERT(all_finite(l1) && all_finite(l6));
            const double err = calc_nmse(l1, l6);
            printf("  %s zero-init NMSE(K1, K6) = %.3e\n", ggml_backend_dev_name(dev), err);
            GGML_ASSERT(err < 1e-6);
        }

        // Test 2: Distinct loop specialization K=1..6 and CPU vs GPU parity
        printf("[Test 2] Per-loop specialization & CPU/GPU parity across K=1..6...\n");
        std::vector<std::vector<float>> cpu_by_k(7);
        for (uint32_t k = 1; k <= 6; ++k) {
            auto g_cpu = make_qwen35_fixture(k, "per_loop");
            auto [m_cpu, c_cpu] = load_fixture(g_cpu.get(), cpu_dev);
            if (k > 1) {
                llama_adapter_lora * a_cpu = llama_adapter_lora_init(m_cpu.get(), spec_path.c_str());
                GGML_ASSERT(a_cpu != nullptr);
                float s = 1.0f;
                GGML_ASSERT(llama_set_adapters_lora(c_cpu.get(), &a_cpu, 1, &s) == 0);
            }
            cpu_by_k[k] = eval_logits_batch(m_cpu.get(), c_cpu.get(), tokens);
            GGML_ASSERT(all_finite(cpu_by_k[k]));

            if (k >= 2) {
                const double diff = calc_nmse(cpu_by_k[k - 1], cpu_by_k[k]);
                printf("  K=%u vs K=%u delta NMSE = %.3e\n", k, k - 1, diff);
                GGML_ASSERT(diff > 1e-4);
            }

            if (gpu_dev != nullptr) {
                auto g_gpu = make_qwen35_fixture(k, "per_loop");
                auto [m_gpu, c_gpu] = load_fixture(g_gpu.get(), gpu_dev);
                if (k > 1) {
                    llama_adapter_lora * a_gpu = llama_adapter_lora_init(m_gpu.get(), spec_path.c_str());
                    GGML_ASSERT(a_gpu != nullptr);
                    float s = 1.0f;
                    GGML_ASSERT(llama_set_adapters_lora(c_gpu.get(), &a_gpu, 1, &s) == 0);
                }
                auto l_gpu = eval_logits_batch(m_gpu.get(), c_gpu.get(), tokens);
                GGML_ASSERT(all_finite(l_gpu));
                const double parity = calc_nmse(cpu_by_k[k], l_gpu);
                printf("  K=%u CPU vs %s NMSE = %.3e\n", k, ggml_backend_dev_name(gpu_dev), parity);
                GGML_ASSERT(parity < 1e-4);
            }
        }

        // Test 3: Prefill vs token-by-token decode parity at K=6 with per-loop LoRA + bridges
        printf("[Test 3] Prefill vs token-by-token parity at K=6...\n");
        {
            ggml_backend_dev_t dev = gpu_dev != nullptr ? gpu_dev : cpu_dev;
            auto g6 = make_qwen35_fixture(6, "per_loop");
            auto [m6, c6] = load_fixture(g6.get(), dev);
            llama_adapter_lora * a6 = llama_adapter_lora_init(m6.get(), spec_path.c_str());
            GGML_ASSERT(a6 != nullptr);
            float s = 1.0f;
            GGML_ASSERT(llama_set_adapters_lora(c6.get(), &a6, 1, &s) == 0);

            auto l_batch = eval_logits_batch(m6.get(), c6.get(), tokens);
            auto l_step  = eval_logits_step_by_step(m6.get(), c6.get(), tokens);
            const double err = calc_nmse(l_batch, l_step);
            printf("  %s prefill vs step-by-step NMSE = %.3e\n", ggml_backend_dev_name(dev), err);
            GGML_ASSERT(err < 1e-4);
        }

        // Test 4: last_loop low-VRAM KV cache mode at K=6
        printf("[Test 4] last_loop low-VRAM KV mode check at K=6...\n");
        {
            auto g_cpu = make_qwen35_fixture(6, "last_loop");
            auto [m_cpu, c_cpu] = load_fixture(g_cpu.get(), cpu_dev);
            llama_adapter_lora * a_cpu = llama_adapter_lora_init(m_cpu.get(), spec_path.c_str());
            GGML_ASSERT(a_cpu != nullptr);
            float s = 1.0f;
            GGML_ASSERT(llama_set_adapters_lora(c_cpu.get(), &a_cpu, 1, &s) == 0);
            auto l_cpu = eval_logits_batch(m_cpu.get(), c_cpu.get(), tokens);
            GGML_ASSERT(all_finite(l_cpu));
            const double diff_vs_k1 = calc_nmse(cpu_by_k[1], l_cpu);
            printf("  last_loop K6 vs K1 NMSE = %.3e\n", diff_vs_k1);
            GGML_ASSERT(diff_vs_k1 > 1e-4);

            if (gpu_dev != nullptr) {
                auto g_gpu = make_qwen35_fixture(6, "last_loop");
                auto [m_gpu, c_gpu] = load_fixture(g_gpu.get(), gpu_dev);
                llama_adapter_lora * a_gpu = llama_adapter_lora_init(m_gpu.get(), spec_path.c_str());
                GGML_ASSERT(a_gpu != nullptr);
                GGML_ASSERT(llama_set_adapters_lora(c_gpu.get(), &a_gpu, 1, &s) == 0);
                auto l_gpu = eval_logits_batch(m_gpu.get(), c_gpu.get(), tokens);
                GGML_ASSERT(all_finite(l_gpu));
                const double parity = calc_nmse(l_cpu, l_gpu);
                printf("  last_loop CPU vs %s NMSE = %.3e\n", ggml_backend_dev_name(gpu_dev), parity);
                GGML_ASSERT(parity < 1e-4);
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    printf("All Qwen3.5 loop tests PASSED!\n");
    return 0;
}
