// Host unit tests for src/hy4/hy4_engram_ram.hpp: CLI/env mode parsing, RAM headroom budget check,
// and sequential pread anonymous-RAM copy across non-page-aligned offsets and multiple 4 MiB chunks.
#include "../src/hy4/hy4_engram_ram.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    // 1. parse_engram_ram_mode: valid and invalid values
    CHECK(hy4::parse_engram_ram_mode("0") == hy4::kEgRamOff);
    CHECK(hy4::parse_engram_ram_mode("off") == hy4::kEgRamOff);
    CHECK(hy4::parse_engram_ram_mode(" OFF ") == hy4::kEgRamOff);
    CHECK(hy4::parse_engram_ram_mode("false") == hy4::kEgRamOff);
    CHECK(hy4::parse_engram_ram_mode("no") == hy4::kEgRamOff);

    CHECK(hy4::parse_engram_ram_mode("1") == hy4::kEgRamOn);
    CHECK(hy4::parse_engram_ram_mode("on") == hy4::kEgRamOn);
    CHECK(hy4::parse_engram_ram_mode("ON") == hy4::kEgRamOn);
    CHECK(hy4::parse_engram_ram_mode("true") == hy4::kEgRamOn);
    CHECK(hy4::parse_engram_ram_mode("yes") == hy4::kEgRamOn);
    CHECK(hy4::parse_engram_ram_mode("ram") == hy4::kEgRamOn);

    CHECK(hy4::parse_engram_ram_mode("2") == hy4::kEgRamLock);
    CHECK(hy4::parse_engram_ram_mode("lock") == hy4::kEgRamLock);
    CHECK(hy4::parse_engram_ram_mode("MLOCK") == hy4::kEgRamLock);

    CHECK(hy4::parse_engram_ram_mode(nullptr) == -1);
    CHECK(hy4::parse_engram_ram_mode("") == -1);
    CHECK(hy4::parse_engram_ram_mode("   ") == -1);
    CHECK(hy4::parse_engram_ram_mode("maybe") == -1);
    CHECK(hy4::parse_engram_ram_mode("3") == -1);

    // 2. resolve_engram_ram_mode precedence: CLI overrides env; invalid/unset defaults to off
    CHECK(hy4::resolve_engram_ram_mode(-1, nullptr) == hy4::kEgRamOff);
    CHECK(hy4::resolve_engram_ram_mode(-1, "invalid") == hy4::kEgRamOff);
    CHECK(hy4::resolve_engram_ram_mode(-1, "1") == hy4::kEgRamOn);
    CHECK(hy4::resolve_engram_ram_mode(-1, "lock") == hy4::kEgRamLock);
    CHECK(hy4::resolve_engram_ram_mode(hy4::kEgRamOff, "lock") == hy4::kEgRamOff);
    CHECK(hy4::resolve_engram_ram_mode(hy4::kEgRamLock, "0") == hy4::kEgRamLock);

    // 3. check_engram_ram_budget
    std::string note;
    const uint64_t GiB = uint64_t{1} << 30;
    CHECK(hy4::check_engram_ram_budget(60 * GiB, 70 * GiB, 4.0, false, note));
    CHECK(note.empty());

    CHECK(!hy4::check_engram_ram_budget(60 * GiB, 63 * GiB, 4.0, false, note));
    CHECK(note.find("HY4_EG_RAM_FORCE=1") != std::string::npos);

    CHECK(hy4::check_engram_ram_budget(60 * GiB, 63 * GiB, 4.0, true, note));
    CHECK(hy4::check_engram_ram_budget(60 * GiB, 0, 4.0, false, note));

    // 4. load_range_to_ram over non-page-aligned offset & >16 MiB (multiple 4 MiB pread chunks)
    const size_t total_size = size_t{19} * 1024 * 1024 + 54321;
    const uint64_t slice_off = 12345;
    const size_t slice_bytes = size_t{17} * 1024 * 1024 + 777;
    std::vector<uint8_t> data(total_size);
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < total_size; ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        data[i] = (uint8_t) (rng >> 56);
    }

#if !defined(_WIN32)
    const std::string tmp_path = "/tmp/hy4_engram_ram_test_" + std::to_string((long long) ::getpid()) + ".bin";
#else
    const std::string tmp_path = "hy4_engram_ram_test.bin";
#endif
    {
        std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
        CHECK(f != nullptr);
        CHECK(std::fwrite(data.data(), 1, data.size(), f) == data.size());
        CHECK(std::fclose(f) == 0);
    }

    size_t mapped_bytes = 0;
    const uint8_t* mapped = mm3::os::map_file(tmp_path, mapped_bytes);
    CHECK(mapped != nullptr);
    CHECK(mapped_bytes == total_size);

    hy4::AnonBuf buf;
    double elapsed = -1.0;
    CHECK(hy4::load_range_to_ram(tmp_path, slice_off, slice_bytes, mapped + slice_off, false,
                                 buf, "test_slice", note, &elapsed));
    CHECK(buf.ptr != nullptr);
    CHECK(buf.bytes == slice_bytes);
    CHECK(elapsed >= 0.0);
#if !defined(_WIN32)
    CHECK(buf.used_pread);
#endif
    CHECK(!buf.locked);
    CHECK(std::memcmp(buf.ptr, data.data() + slice_off, slice_bytes) == 0);

    // 5. want_lock=true either locks or gracefully keeps unlocked anonymous RAM under tight RLIMIT_MEMLOCK
    hy4::AnonBuf lock_buf;
    const size_t lock_bytes = size_t{1} << 20;
    CHECK(hy4::load_range_to_ram(tmp_path, slice_off, lock_bytes, mapped + slice_off, true,
                                 lock_buf, "test_lock", note));
    CHECK(lock_buf.ptr != nullptr);
    CHECK(lock_buf.bytes == lock_bytes);
    CHECK(std::memcmp(lock_buf.ptr, data.data() + slice_off, lock_bytes) == 0);

    buf.release();
    CHECK(buf.ptr == nullptr);
    CHECK(buf.bytes == 0);
    CHECK(!buf.locked);
    CHECK(!buf.used_pread);

    lock_buf.release();
    (void) std::remove(tmp_path.c_str());

    // 6. load_spans_to_ram: multi-file sorting, contiguous pread coalescing, non-contiguous gaps & keep_writable
    const size_t f0_size = size_t{9} * 1024 * 1024 + 4096;
    const size_t f1_size = size_t{5} * 1024 * 1024 + 8192;
    std::vector<uint8_t> d0(f0_size), d1(f1_size);
    for (size_t i = 0; i < f0_size; ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        d0[i] = (uint8_t) (rng >> 56);
    }
    for (size_t i = 0; i < f1_size; ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        d1[i] = (uint8_t) (rng >> 56);
    }
#if !defined(_WIN32)
    const std::string p0 = "/tmp/hy4_spans_0_" + std::to_string((long long) ::getpid()) + ".bin";
    const std::string p1 = "/tmp/hy4_spans_1_" + std::to_string((long long) ::getpid()) + ".bin";
#else
    const std::string p0 = "hy4_spans_0.bin";
    const std::string p1 = "hy4_spans_1.bin";
#endif
    {
        std::FILE* f = std::fopen(p0.c_str(), "wb");
        CHECK(f != nullptr && std::fwrite(d0.data(), 1, d0.size(), f) == d0.size() && std::fclose(f) == 0);
    }
    {
        std::FILE* f = std::fopen(p1.c_str(), "wb");
        CHECK(f != nullptr && std::fwrite(d1.data(), 1, d1.size(), f) == d1.size() && std::fclose(f) == 0);
    }
    size_t m0_bytes = 0, m1_bytes = 0;
    const uint8_t* m0 = mm3::os::map_file(p0, m0_bytes);
    const uint8_t* m1 = mm3::os::map_file(p1, m1_bytes);
    CHECK(m0 != nullptr && m1 != nullptr);

    // Out-of-order spans with both contiguous runs (must coalesce) and gaps across both files.
    const size_t sp_a_off = 3333, sp_a_len = 2 * 1024 * 1024 + 555;
    const size_t sp_b_off = sp_a_off + sp_a_len, sp_b_len = 3 * 1024 * 1024 + 111;   // contiguous after sp_a
    const size_t sp_c_off = sp_b_off + sp_b_len + 123456, sp_c_len = 1500000;      // gap after sp_b
    const size_t sp_d_off = 4096, sp_d_len = 4 * 1024 * 1024 + 777;                // in file 1
    std::vector<hy4::RamSpan> spans = {
        {1, sp_d_off, sp_d_len, 0, m1 + sp_d_off, 2, 0, 5},
        {0, sp_c_off, sp_c_len, 0, m0 + sp_c_off, 1, 2, 9},
        {0, sp_b_off, sp_b_len, 0, m0 + sp_b_off, 0, 1, 3},
        {0, sp_a_off, sp_a_len, 0, m0 + sp_a_off, 0, 0, 3},
    };
    hy4::AnonBuf span_buf;
    double span_elapsed = -1.0;
    CHECK(hy4::load_spans_to_ram({p0, p1}, spans, false, false, span_buf, "test_spans", note, &span_elapsed));
    CHECK(span_buf.ptr != nullptr);
    CHECK(span_buf.bytes == sp_a_len + sp_b_len + sp_c_len + sp_d_len);
    CHECK(span_elapsed >= 0.0);
#if !defined(_WIN32)
    CHECK(span_buf.used_pread);
#endif
    for (const hy4::RamSpan& sp : spans) {
        const std::vector<uint8_t>& src_vec = (sp.file == 0) ? d0 : d1;
        CHECK(std::memcmp(span_buf.ptr + sp.dst_off, src_vec.data() + sp.file_off, sp.bytes) == 0);
    }
    span_buf.release();

    // Writable buffer test (used when HY4_HOT=1 swaps evicted VRAM experts back into RAM)
    CHECK(hy4::load_spans_to_ram({p0, p1}, spans, false, true, span_buf, "test_spans_rw", note));
    CHECK(span_buf.ptr != nullptr);
    ((uint8_t*) span_buf.ptr)[0] ^= 0xff;
    span_buf.release();
    CHECK(span_buf.ptr == nullptr && span_buf.bytes == 0);

    (void) std::remove(p0.c_str());
    (void) std::remove(p1.c_str());

    std::fprintf(stderr, "hy4_engram_ram_test: ok\n");
    return 0;
}
