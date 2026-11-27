// src/core/glm_model_test.cpp - the glm5-next runner vs the oracle, and streaming vs batched.
//
// Four gates (docs/GLM5-FLASH.md Phase 3):
//
//   1. BATCHED VS THE ORACLE: forward(prompt) must reproduce ref/glm.py's teacher-forced logits
//      (the last column of the result_output dump) - the whole model through the runner.
//   2. STREAMING VS BATCHED: single-token calls on a persistent state must produce the SAME logits
//      as the batched call.  This is the gate that matters for the engine: decode IS streaming, and
//      the runner runs the identical per-token steps either way, so the tolerance is tight.
//   3. EVERY PREFIX ARGMAX: each teacher-forced prefix's argmax must match the oracle's - the
//      sharpest end-to-end check, because it flips on any seam that diverges enough to reorder the
//      top logits (each prefix is its OWN sequence: the runner is reset before it).
//   4. GREEDY CONTINUATION: argmax-stepping the runner (state persisting) must follow the oracle's
//      greedy chain for the CURRENT dump - computed at run time from the dump's own columns,
//      never a constant (a Phase-1 chain hardcoded here went stale when the oracle was repaired).
#include "strata/core/glm_model.hpp"
#include "strata/kernels/cpu/glm_lane_row4.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_model_test: %s\n", what.c_str());
        std::exit(1);
    }
}

// the oracle's dump: (vocab, T) ggml order - element (v, t) at v + vocab*t
std::vector<float> load_logits(const std::string& path, size_t vocab, size_t T) {
    std::vector<float> v(vocab * T);
    FILE* f = std::fopen(path.c_str(), "rb");
    require(f != nullptr, "cannot open " + path);
    require(std::fread(v.data(), sizeof(float), v.size(), f) == v.size(), "short read on " + path);
    std::fclose(f);
    return v;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

double max_abs(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

size_t argmax(const std::vector<float>& v) {
    size_t am = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[am]) am = i;
    return am;
}

// CPU-only bit-exact parity test (--cpu-q4x), executed BEFORE any GPU/HIP init:
// verifies TG-B2 (k_iq3_xxs row4, k_iq3_s row4, native_down_rows_split shadowing fix) and
// TG-B1 (Down-only Q4X vpshufb AVX2 for IQ3_XXS and IQ3_S, nt=1..5) bit-for-bit against ggml-cpu.
int run_cpu_q4x_tests() {
    namespace kc = strata::kernels::cpu;
    ggml_cpu_init();
    int fails = 0, checks = 0;
    double worst_abs = 0.0;
    auto check_eq = [&](const float* got, const float* ref, int n, const std::string& label) {
        ++checks;
        double ma = 0.0;
        for (int i = 0; i < n; ++i) ma = std::max(ma, (double) std::fabs(got[i] - ref[i]));
        worst_abs = std::max(worst_abs, ma);
        if (std::memcmp(got, ref, (size_t) n * sizeof(float)) != 0) {
            std::fprintf(stderr, "glm_model_test --cpu-q4x FAIL [%s]: max_abs_diff=%.9g\n", label.c_str(), ma);
            ++fails;
            return false;
        }
        return true;
    };
    auto check_bool = [&](bool ok, const std::string& label) {
        ++checks;
        if (!ok) {
            std::fprintf(stderr, "glm_model_test --cpu-q4x FAIL [%s]\n", label.c_str());
            ++fails;
        }
    };

    const bool has_avx2 = ggml_cpu_has_avx2() != 0 && ggml_cpu_has_fma() != 0 && ggml_cpu_has_f16c() != 0;
    std::printf("glm_model_test --cpu-q4x: avx2=%d fma=%d f16c=%d | row4(IQ3_XXS=%d IQ3_S=%d) | q4x(IQ3_XXS=%d IQ3_S=%d)\n",
                ggml_cpu_has_avx2(), ggml_cpu_has_fma(), ggml_cpu_has_f16c(),
                kc::row4_supported(GGML_TYPE_IQ3_XXS) ? 1 : 0, kc::row4_supported(GGML_TYPE_IQ3_S) ? 1 : 0,
                kc::native_q4x_supported(GGML_TYPE_IQ3_XXS) ? 1 : 0, kc::native_q4x_supported(GGML_TYPE_IQ3_S) ? 1 : 0);
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx2) {
        check_bool(kc::row4_supported(GGML_TYPE_IQ3_XXS), "row4_supported(IQ3_XXS)");
        check_bool(kc::row4_supported(GGML_TYPE_IQ3_S), "row4_supported(IQ3_S)");
        check_bool(kc::native_q4x_supported(GGML_TYPE_IQ3_XXS), "native_q4x_supported(IQ3_XXS)");
        check_bool(kc::native_q4x_supported(GGML_TYPE_IQ3_S), "native_q4x_supported(IQ3_S)");
    }
#endif
    if (!has_avx2 || fails > 0) {
        std::printf("glm_model_test --cpu-q4x: %s (%d checks, %d fails)\n", fails ? "FAIL" : "SKIP (no AVX2)", checks, fails);
        return fails ? 1 : 0;
    }

    auto fill_weights = [](int type, uint8_t* p, size_t bytes, std::mt19937& rng, int fill_mode) {
        std::uniform_int_distribution<int> ub(0, 255);
        std::uniform_real_distribution<float> sc(-0.02f, 0.02f);
        for (size_t i = 0; i < bytes; ++i)
            p[i] = fill_mode == 1 ? 0xFF : fill_mode == 2 ? 0x00 : (uint8_t) ub(rng);
        const size_t bs = ggml_type_size((ggml_type) type);
        for (size_t o = 0; o + bs <= bytes; o += bs) {
            if (type == GGML_TYPE_IQ3_S && (fill_mode == 3 || fill_mode == 4)) {
                // qh[8] + signs[32] + scales[4] at offset 66..110
                std::memset(p + o + 66, fill_mode == 3 ? 0xFF : 0x00, 44);
            }
            const ggml_fp16_t d = ggml_fp32_to_fp16(fill_mode == 1 || fill_mode == 2 ? 0.015f : sc(rng));
            std::memcpy(p + o, &d, sizeof d);
        }
    };
    auto make_q8k = [](int n, std::mt19937& rng, int mode) {
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_K, n));
        if (mode == 2) {
            // Raw extreme -128 / +127 Q8_K bytes (tests vpsignb exactness in row4 kernels)
            const size_t bs = ggml_type_size(GGML_TYPE_Q8_K);
            for (int b = 0; b < n / 256; ++b) {
                uint8_t* p = q.data() + (size_t) b * bs;
                const float d = 0.0117f;
                std::memcpy(p, &d, sizeof d);
                int8_t qs[256];
                for (int j = 0; j < 256; ++j) qs[j] = (rng() & 1u) ? (int8_t) 127 : (int8_t) -128;
                std::memcpy(p + 4, qs, sizeof qs);
                for (int g = 0; g < 16; ++g) {
                    int s = 0;
                    for (int j = 0; j < 16; ++j) s += qs[16 * g + j];
                    const int16_t s16 = (int16_t) s;
                    std::memcpy(p + 260 + 2 * g, &s16, sizeof s16);
                }
            }
            return q;
        }
        std::vector<float> x((size_t) n);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < n; ++i)
            x[(size_t) i] = mode == 1 ? ((rng() & 1u) ? 1.0f : -1.0f) : nd(rng) * (i % 61 == 0 ? 25.0f : 1.0f);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), q.data(), n);
        return q;
    };

    const int types[2] = {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S};
    const int seeds[2] = {20261127, 133742};
    const int widths[3] = {256, 1024, 2048};

    for (int seed : seeds) {
        std::mt19937 rng((uint32_t) seed);
        for (int type : types) {
            const char* tname = ggml_type_name((ggml_type) type);
            const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
            for (int n : widths) {
                const size_t rb = ggml_row_size((ggml_type) type, n);
                const size_t qrb = kc::native_q4x_row_bytes(n);
                constexpr int kRows = 20;
                const int max_fill = type == GGML_TYPE_IQ3_S ? 5 : 3;
                for (int fill_mode = 0; fill_mode < max_fill; ++fill_mode) {
                    std::vector<uint8_t> w(rb * (size_t) kRows);
                    fill_weights(type, w.data(), w.size(), rng, fill_mode);
                    std::vector<uint8_t> qbuf(qrb * (size_t) kRows + 32);
                    uint8_t* qp = (uint8_t*) (((uintptr_t) qbuf.data() + 31) & ~(uintptr_t) 31);
                    kc::native_q4x_convert(type, w.data(), rb, kRows, n, qp);

                    // 1. row4_rows + row4_dot (including raw -128/+127 Q8_K)
                    for (int amode = 0; amode < 3; ++amode) {
                        const auto q8 = make_q8k(n, rng, amode);
                        std::vector<float> ref(kRows);
                        for (int r = 0; r < kRows; ++r)
                            dot(n, &ref[(size_t) r], 0, w.data() + (size_t) r * rb, 0, q8.data(), 0, 1);
                        for (int r0 : {0, 1, 3}) {
                            for (int cnt : {1, 3, 4, 5, 8, 9, 13}) {
                                if (r0 + cnt > kRows) continue;
                                std::vector<float> out(kRows, -9876.5f);
                                kc::row4_rows(type, n, w.data(), rb, q8.data(), out.data(), r0, r0 + cnt);
                                check_eq(out.data() + r0, ref.data() + r0, cnt,
                                         std::string("row4 ") + tname + " n=" + std::to_string(n) +
                                         " fill=" + std::to_string(fill_mode) + " amode=" + std::to_string(amode));
                                bool guard_ok = true;
                                for (int r = 0; r < kRows; ++r)
                                    if ((r < r0 || r >= r0 + cnt) && out[(size_t) r] != -9876.5f) guard_ok = false;
                                check_bool(guard_ok, std::string("row4 guard ") + tname);
                            }
                        }
                        // Scattered 4 rows via row4_dot
                        const int pick[4] = {15, 2, 11, 0};
                        const uint8_t* rows4[4] = {w.data() + pick[0] * rb, w.data() + pick[1] * rb,
                                                   w.data() + pick[2] * rb, w.data() + pick[3] * rb};
                        float out4[4], ref4[4];
                        kc::row4_dot(type, n, rows4, q8.data(), out4);
                        for (int k = 0; k < 4; ++k) ref4[k] = ref[(size_t) pick[k]];
                        check_eq(out4, ref4, 4, std::string("row4_dot ") + tname);
                    }

                    // 2. Q4X bit-exact parity across nt = 1..5, offsets r0 = 0,1,3, counts including non-multiples of 4
                    for (int amode = 0; amode < 2; ++amode) {
                        for (int nt : {1, 2, 3, 4, 5}) {
                            std::vector<std::vector<uint8_t>> acts((size_t) nt);
                            std::vector<const void*> ap((size_t) nt);
                            std::vector<float> ref((size_t) nt * kRows);
                            for (int t = 0; t < nt; ++t) {
                                acts[(size_t) t] = make_q8k(n, rng, amode);
                                ap[(size_t) t] = acts[(size_t) t].data();
                                for (int r = 0; r < kRows; ++r)
                                    dot(n, &ref[(size_t) t * kRows + r], 0, w.data() + (size_t) r * rb, 0, ap[(size_t) t], 0, 1);
                            }
                            for (int r0 : {0, 1, 3}) {
                                for (int cnt : {1, 3, 4, 5, 9, 17}) {
                                    if (r0 + cnt > kRows) continue;
                                    std::vector<float> got((size_t) nt * kRows, -5432.25f);
                                    std::vector<float*> op((size_t) nt);
                                    for (int t = 0; t < nt; ++t) op[(size_t) t] = got.data() + (size_t) t * kRows;
                                    kc::native_q4x_down_rows(type, qp, qrb, n, ap.data(), nt, op.data(), r0, r0 + cnt);
                                    for (int t = 0; t < nt; ++t) {
                                        check_eq(got.data() + (size_t) t * kRows + r0, ref.data() + (size_t) t * kRows + r0, cnt,
                                                 std::string("q4x ") + tname + " n=" + std::to_string(n) +
                                                 " nt=" + std::to_string(nt) + " fill=" + std::to_string(fill_mode));
                                        bool guard_ok = true;
                                        for (int r = 0; r < kRows; ++r)
                                            if ((r < r0 || r >= r0 + cnt) && got[(size_t) t * kRows + r] != -5432.25f)
                                                guard_ok = false;
                                        check_bool(guard_ok, std::string("q4x guard ") + tname);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 3. Maya-M real down shape (n_ff = 2048, 64 rows): native_down_rows_split (row4 ON vs OFF) vs Q4X vs ggml
            kc::NativeFmt nf{};
            std::string err;
            check_bool(kc::native_fmt(GGML_TYPE_IQ2_S, type, 4096, 2048, nf, err),
                       std::string("native_fmt IQ2_S/") + tname + ": " + err);
            const int rows = 64;
            std::vector<uint8_t> dn(nf.d_row * (size_t) rows);
            fill_weights(type, dn.data(), dn.size(), rng, 0);
            const size_t qrb = kc::native_q4x_row_bytes(2048);
            std::vector<uint8_t> qbuf(qrb * (size_t) rows + 32);
            uint8_t* qp = (uint8_t*) (((uintptr_t) qbuf.data() + 31) & ~(uintptr_t) 31);
            kc::native_q4x_convert(type, dn.data(), nf.d_row, rows, 2048, qp);
            const auto hq = make_q8k(2048, rng, 0);
            const void* hp[1] = {hq.data()};
            std::vector<float> o_off(rows, 0.f), o_on(rows, 0.f), o_q4(rows, 0.f), o_ref(rows, 0.f);
            float* p_off[1] = {o_off.data()};
            float* p_on[1] = {o_on.data()};
            float* p_q4[1] = {o_q4.data()};
            for (int r = 0; r < rows; ++r)
                dot(2048, &o_ref[(size_t) r], 0, dn.data() + (size_t) r * nf.d_row, 0, hq.data(), 0, 1);
            kc::lane_row4_override(0);
            kc::native_down_rows_split(nf, dn.data(), hp, 1, p_off, 0, rows);
            kc::lane_row4_override(1);
            kc::native_down_rows_split(nf, dn.data(), hp, 1, p_on, 0, rows);
            kc::lane_row4_override(-1);
            kc::native_q4x_down_rows(type, qp, qrb, 2048, hp, 1, p_q4, 0, rows);
            check_eq(o_off.data(), o_ref.data(), rows, std::string("split_off ") + tname);
            check_eq(o_on.data(), o_ref.data(), rows, std::string("split_on ") + tname);
            check_eq(o_q4.data(), o_ref.data(), rows, std::string("split_q4x ") + tname);
        }
    }

    std::printf("glm_model_test --cpu-q4x: %s (%d checks, %d fails, max_abs_diff=%.3e)\n",
                fails ? "FAIL" : "PASS", checks, fails, worst_abs);
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--cpu-q4x") {
        return run_cpu_q4x_tests();
    }
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_model_test: run with --selftest (the CTest form)\n");
        return 2;
    }
    const std::string gguf = argc > 2 ? argv[2] : "data/glm5-synth.gguf";
    const std::string dumps = argc > 3 ? argv[3] : "data/glm5-synth-dumps";
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    // the data paths are relative to the repo root (the CTest form runs there): from the build folder it ended with
    // no message, 0xC0000409 on Windows (#54)
    if (std::FILE* f = std::fopen(gguf.c_str(), "rb")) {
        std::fclose(f);
    } else {
        std::fprintf(stderr, "glm_model_test: %s not found - run it from the repo root, or pass the file\n",
                     gguf.c_str());
        return 2;
    }

    strata::core::Glm5Model model;
    std::string err;
    require(model.load(gguf, 64, err), "load: " + err);
    const size_t V = (size_t) model.geometry().n_vocab;
    const size_t T = prompt.size();
    const std::vector<float> oracle = load_logits(dumps + "/result_output.f32", V, T);

    // ---- 1. batched vs the oracle (last token's logits)
    std::vector<float> batched;
    require(model.forward(prompt, batched, err), "batched forward: " + err);
    const std::vector<float> oracle_last(oracle.begin() + (long) V * (long) (T - 1), oracle.end());
    const double d_batched = max_abs(batched, oracle_last);
    require(d_batched < 2e-3, "batched logits diverge from ref/glm.py (" + std::to_string(d_batched) + ")");

    // ---- 3. every prefix argmax (each prefix its own sequence: reset first)
    for (size_t t = 1; t <= T; ++t) {
        model.reset();
        std::vector<float> pref;
        require(model.forward(std::vector<int32_t>(prompt.begin(), prompt.begin() + (long) t), pref, err),
                "prefix forward: " + err);
        const size_t am = argmax(pref);
        const float* col = oracle.data() + V * (t - 1);
        const size_t om = argmax(std::vector<float>(col, col + V));
        if (am != om)
            std::fprintf(stderr, "dbg prefix %zu: argmax %zu vs %zu, maxdiff %.3e\n", t, am, om,
                         max_abs(pref, std::vector<float>(col, col + V)));
        require(am == om, "prefix " + std::to_string(t) + " argmax differs from the oracle");
    }

    // ---- 2. streaming vs batched: the same per-token steps, so the same values
    model.reset();
    std::vector<float> streamed;
    for (int32_t tok : prompt) {
        std::vector<float> one;
        require(model.forward({tok}, one, err), "streamed forward: " + err);
        streamed = std::move(one);
    }
    const double d_stream = max_abs(streamed, batched);
    require(d_stream < 1e-5, "streamed and batched logits differ (" + std::to_string(d_stream) + ")");

    // ---- 4. full-sequence parity at 12 tokens (prompt 1..8 + the oracle's own greedy chain) and
    // streaming continuation.  The expected chain is DERIVED from the committed 12-token oracle
    // dump, never hardcoded - a hardcoded chain goes stale the moment the oracle gains a fix
    // (this exact bug: the Phase-1 chain [385, 93, 505, 203] predated the conv-einsum repair and
    // silently poisoned this gate).  The runner's forward returns the LAST token's logits, so
    // per-column parity means one forward per prefix (78 cheap steps total), and the streaming
    // check chains one 8-token prefill plus the 4 chain tokens singly from the same state.
    const std::string dumps12 = argc > 4 ? argv[4] : "data/glm5-synth-dumps12";
    const size_t T12 = 12;
    const std::vector<float> oracle12 = load_logits(dumps12 + "/result_output.f32", V, T12);
    const std::vector<int32_t> tokens12 = {1, 2, 3, 4, 5, 6, 7, 8, 98, 162, 57, 401};
    std::vector<int32_t> greedy_expected;
    for (size_t t = T - 1; t < T12; ++t) {
        const float* col = oracle12.data() + V * t;
        greedy_expected.push_back((int32_t) argmax(std::vector<float>(col, col + V)));
    }

    double worst12 = 0.0;
    std::vector<int32_t> greedy;
    for (size_t t = 1; t <= T12; ++t) {
        model.reset();
        std::vector<float> lg;
        require(model.forward(std::vector<int32_t>(tokens12.begin(), tokens12.begin() + (long) t), lg, err),
                "12-token forward: " + err);
        worst12 = std::max(worst12, max_abs(lg.data(), oracle12.data() + V * (t - 1), V));
        if (t >= T)
            greedy.push_back((int32_t) argmax(lg));   // columns 8..11: the greedy chain
    }
    require(worst12 < 2e-3, "12-token logits diverge from the oracle (" + std::to_string(worst12) + ")");
    require(greedy == greedy_expected, "the runner's greedy chain diverges from the oracle's");

    // streaming continuation: one 8-token prefill, then the chain fed singly from the same state;
    // every continuation logit vs the oracle's columns 8..11 - the persistent-state check at the
    // prefill-to-decode boundary (positions 8..11 have never been covered by any other gate)
    model.reset();
    std::vector<float> cont;
    require(model.forward(prompt, cont, err), "continuation prefill: " + err);
    double worst_cont = 0.0;
    for (size_t t = T; t < T12; ++t) {
        require(model.forward({tokens12[t]}, cont, err), "continuation forward: " + err);
        worst_cont = std::max(worst_cont, max_abs(cont.data(), oracle12.data() + V * t, V));
    }
    require(worst_cont < 2e-3, "continuation logits diverge from the oracle (" + std::to_string(worst_cont) + ")");

    // ---- 5. STATE DETERMINISM (second opinion): forward(A), reset, forward(B), reset,
    // forward(A) again - the third run must reproduce the first BITWISE.  Anything stateful
    // leaking across sequences (an unwashed KDA S, a stale DSA pool row) shows up here.
    const std::vector<int32_t> seqB = {11, 12, 13, 14, 15, 16, 17, 18};
    std::vector<float> runA1, runB, runA2;
    model.reset();   // gate 4's greedy loop leaves the state at position 12; start clean
    require(model.forward(prompt, runA1, err), "state A1: " + err);
    model.reset();
    require(model.forward(seqB, runB, err), "state B: " + err);
    model.reset();
    require(model.forward(prompt, runA2, err), "state A2: " + err);
    require(runA1 == runA2, "state determinism: A-after-B differs bitwise from A (max "
                                + std::to_string(max_abs(runA1, runA2)) + ")");

    // ---- 6. MUTATION SENSITIVITY (second opinion): a corrupted weight MUST move the logits,
    // or gate 1 could pass against a wrong model.  Copy the GGUF, overwrite 4 KB inside a
    // routed expert slab, reload, forward - the max delta must be far above f32 noise.
    {
        const std::string corrupt_path = gguf + ".mutation.tmp";
        FILE* src = std::fopen(gguf.c_str(), "rb");
        require(src != nullptr, "mutation: cannot open " + gguf);
        std::fseek(src, 0, SEEK_END);
        long size = std::ftell(src);
        std::fseek(src, 0, SEEK_SET);
        std::vector<char> blob((size_t) size);
        require(std::fread(blob.data(), 1, blob.size(), src) == blob.size(), "mutation: short read");
        std::fclose(src);
        // find the expert slab by scanning for its float payload: the fixture's blk.1
        // ffn_gate_exps starts after the metadata; corrupt a slice in the middle of the file
        const long at = size / 2;                      // any mid-file weight bytes
        float* f = (float*) (blob.data() + at);
        for (int i = 0; i < 1024; ++i) f[i] = f[i] * -4.0f + 3.0f;
        FILE* dst = std::fopen(corrupt_path.c_str(), "wb");
        require(dst != nullptr, "mutation: cannot write copy");
        require(std::fwrite(blob.data(), 1, blob.size(), dst) == blob.size(), "mutation: short write");
        std::fclose(dst);

        strata::core::Glm5Model corrupt;
        require(corrupt.load(corrupt_path, 64, err), "mutation load: " + err);
        std::vector<float> mutated;
        require(corrupt.forward(prompt, mutated, err), "mutation forward: " + err);
        std::remove(corrupt_path.c_str());
        const double d_mut = max_abs(mutated, batched);
        require(d_mut > 1e-2, "mutation was NOT detected by the logits (max delta "
                                  + std::to_string(d_mut) + " - gate 1 cannot catch a wrong model)");
        std::printf("mutation sensitivity: %.3e\n", d_mut);
    }

    std::printf("glm_model_test: PASS (batched-8 %.2e, streamed-8 %.2e, 12-token %.2e, continuation %.2e)\n",
                d_batched, d_stream, worst12, worst_cont);
    return 0;
}
