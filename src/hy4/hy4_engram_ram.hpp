// src/hy4/hy4_engram_ram.hpp - optional anonymous-RAM loader for maya-hy4's disk-backed Engram & token embeddings.
//
// Designed for HDD-backed GGUF shards under swapless Linux hosts with Transparent Huge Pages [always]:
//   - Off by default (HY4_EG_RAM=0 / --engram-ram off) so small-RAM hosts never allocate tens of GB unexpectedly.
//   - When enabled (HY4_EG_RAM=1|on or 2|lock / --engram-ram on|lock), performs a budget check against MemAvailable /
//     cgroup v2 limits + headroom, then streams the file range sequentially via 4 MiB pread() + POSIX_FADV_DONTNEED
//     into anonymous mmap() memory marked MADV_NOHUGEPAGE (preventing Direct Compaction freezes and avoiding Linux
//     page-cache double buffering or file-cache eviction).
#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../mm3/mm3_os.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace hy4 {

enum EngramRamMode {
    kEgRamOff = 0,
    kEgRamOn = 1,
    kEgRamLock = 2,
};

inline int parse_engram_ram_mode(const char* s) {
    if (s == nullptr) return -1;
    std::string v;
    for (const unsigned char* p = (const unsigned char*) s; *p; ++p) {
        if (!std::isspace(*p)) v.push_back((char) std::tolower(*p));
    }
    if (v.empty()) return -1;
    if (v == "0" || v == "off" || v == "false" || v == "no") return kEgRamOff;
    if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "ram") return kEgRamOn;
    if (v == "2" || v == "lock" || v == "mlock") return kEgRamLock;
    return -1;
}

inline int resolve_engram_ram_mode(int cli_mode, const char* env_val = std::getenv("HY4_EG_RAM")) {
    if (cli_mode >= 0) return cli_mode <= kEgRamLock ? cli_mode : kEgRamLock;
    const int parsed = parse_engram_ram_mode(env_val);
    return parsed >= 0 ? parsed : kEgRamOff;
}

inline bool check_engram_ram_budget(uint64_t need_bytes, uint64_t avail_bytes, double headroom_gb, bool force,
                                    std::string& note) {
    note.clear();
    if (force || avail_bytes == 0) return true;
    const double headroom_clamp = std::max(0.0, headroom_gb);
    const double headroom_bytes = headroom_clamp * 1073741824.0;
    if ((double) avail_bytes >= (double) need_bytes + headroom_bytes) return true;
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "need %.2f GiB + %.1f GiB headroom, available %.2f GiB (set HY4_EG_RAM_FORCE=1 to override)",
                  (double) need_bytes / 1073741824.0, headroom_clamp, (double) avail_bytes / 1073741824.0);
    note = buf;
    return false;
}

struct AnonBuf {
    uint8_t* ptr = nullptr;
    size_t bytes = 0;
    bool locked = false;
    bool used_pread = false;

    AnonBuf() = default;
    AnonBuf(const AnonBuf&) = delete;
    AnonBuf& operator=(const AnonBuf&) = delete;
    AnonBuf(AnonBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes), locked(o.locked), used_pread(o.used_pread) {
        o.ptr = nullptr;
        o.bytes = 0;
        o.locked = false;
        o.used_pread = false;
    }
    AnonBuf& operator=(AnonBuf&& o) noexcept {
        if (this != &o) {
            release();
            ptr = o.ptr;
            bytes = o.bytes;
            locked = o.locked;
            used_pread = o.used_pread;
            o.ptr = nullptr;
            o.bytes = 0;
            o.locked = false;
            o.used_pread = false;
        }
        return *this;
    }
    ~AnonBuf() { release(); }

    void release() {
        if (ptr != nullptr) {
#if !defined(_WIN32)
            if (locked) (void) ::munlock(ptr, bytes);
            (void) ::munmap(ptr, bytes);
#else
            std::free(ptr);
#endif
        }
        ptr = nullptr;
        bytes = 0;
        locked = false;
        used_pread = false;
    }
};

inline bool load_range_to_ram(const std::string& path, uint64_t off, size_t bytes, const uint8_t* mmap_fallback,
                              bool want_lock, AnonBuf& dst, const char* label, std::string& note,
                              double* elapsed_s = nullptr) {
    dst.release();
    note.clear();
    if (elapsed_s) *elapsed_s = 0.0;
    if (bytes == 0) return true;

    const auto t0 = std::chrono::steady_clock::now();

#if !defined(_WIN32)
    void* mem = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        char ebuf[192];
        std::snprintf(ebuf, sizeof ebuf, "mmap(%zu bytes) failed: %s", bytes, std::strerror(errno));
        note = ebuf;
        return false;
    }
#if defined(MADV_NOHUGEPAGE)
    (void) ::madvise(mem, bytes, MADV_NOHUGEPAGE);
#endif
    dst.ptr = (uint8_t*) mem;
    dst.bytes = bytes;

    if (!path.empty()) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
#if defined(POSIX_FADV_SEQUENTIAL)
            (void) ::posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_SEQUENTIAL);
#endif
            constexpr size_t kChunk = size_t{4} << 20;
            constexpr size_t kDropStep = size_t{64} << 20;
            constexpr size_t kLogStep = size_t{4} << 30;
            size_t done = 0, dropped = 0, next_log = kLogStep;
            while (done < bytes) {
                const size_t want = std::min(kChunk, bytes - done);
                const ssize_t r = ::pread(fd, dst.ptr + done, want, (off_t) (off + done));
                if (r < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                if (r == 0) break;
                done += (size_t) r;
                if (done - dropped >= kDropStep) {
#if defined(POSIX_FADV_DONTNEED)
                    (void) ::posix_fadvise(fd, (off_t) (off + dropped), (off_t) (done - dropped), POSIX_FADV_DONTNEED);
#endif
                    dropped = done;
                }
                if (bytes >= (size_t{256} << 20) && done >= next_log && done < bytes) {
                    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    std::fprintf(stderr, "maya-hy4: %s RAM load: %.2f / %.2f GB (%.0f MB/s)\n",
                                 label ? label : "Engram", (double) done / 1e9, (double) bytes / 1e9,
                                 ((double) done / 1e6) / std::max(1e-6, dt));
                    next_log += kLogStep;
                }
            }
            if (done > dropped) {
#if defined(POSIX_FADV_DONTNEED)
                (void) ::posix_fadvise(fd, (off_t) (off + dropped), (off_t) (done - dropped), POSIX_FADV_DONTNEED);
#endif
            }
            (void) ::close(fd);
            if (done == bytes) dst.used_pread = true;
        }
    }

    if (!dst.used_pread) {
        if (mmap_fallback != nullptr) {
            std::memcpy(dst.ptr, mmap_fallback, bytes);
        } else {
            note = "pread failed and no mmap source available";
            dst.release();
            return false;
        }
    }

    if (mmap_fallback != nullptr) mm3::os::drop_mapped(mmap_fallback, bytes);

    if (want_lock) {
        constexpr size_t kLockChunk = size_t{256} << 20;
        size_t locked_bytes = 0;
        bool lock_ok = true;
        int lock_err = 0;
        while (locked_bytes < bytes) {
            const size_t n = std::min(kLockChunk, bytes - locked_bytes);
            if (::mlock(dst.ptr + locked_bytes, n) != 0) {
                lock_err = errno;
                lock_ok = false;
                break;
            }
            locked_bytes += n;
        }
        if (lock_ok) {
            dst.locked = true;
        } else {
            if (locked_bytes > 0) (void) ::munlock(dst.ptr, locked_bytes);
            dst.locked = false;
            char lbuf[192];
            std::snprintf(lbuf, sizeof lbuf, "mlock refused (%s); keeping unlocked anonymous RAM",
                          std::strerror(lock_err));
            note = lbuf;
        }
    }

    (void) ::mprotect(dst.ptr, bytes, PROT_READ);
#else
    (void) path;
    (void) off;
    (void) want_lock;
    (void) label;
    uint8_t* mem = (uint8_t*) std::malloc(bytes);
    if (mem == nullptr) {
        note = "malloc failed";
        return false;
    }
    if (mmap_fallback == nullptr) {
        std::free(mem);
        note = "no mmap source available";
        return false;
    }
    std::memcpy(mem, mmap_fallback, bytes);
    dst.ptr = mem;
    dst.bytes = bytes;
    dst.locked = false;
    dst.used_pread = false;
#endif

    if (elapsed_s) {
        *elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    return true;
}

}  // namespace hy4
