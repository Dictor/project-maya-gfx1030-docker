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

    std::fprintf(stderr, "hy4_engram_ram_test: ok\n");
    return 0;
}
