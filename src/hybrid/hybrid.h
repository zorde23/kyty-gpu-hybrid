#ifndef EMULATOR_SRC_HYBRID_HYBRID_H_
#define EMULATOR_SRC_HYBRID_HYBRID_H_

#include <atomic>
#include <cstdint>

// The AnyPS5 executable hosts Kyty's GPU as kyty_gpu.dll (the hybrid). The executable owns guest memory,
// threads, the exception handler and events; Kyty only tracks and draws. Aps5Host is the table of the
// executable's callbacks that kyty_gpu_init hands over (version 2).
struct Aps5Host {
	uint32_t version;
	// tracker: 0 = no tracking protection, 1 = read-only, 2 = no access; per 4 KiB page of donated memory.
	bool (*protect_host)(uint64_t address, uint64_t bytes, uint32_t tracker);
	// The executable's own export of a guest library NID (libkernel first), SysV ABI, or 0.
	uint64_t (*resolve)(const char* nid);
};

namespace Hybrid {

// KYTY_FAULT_STATS: time this thread spent in the fault handler's slow parts (protect_host, buffer
// readback, the texture cache lock), read before and after each stage of a fault. Dry.
struct FaultStageTime {
	uint64_t protect_ns = 0, protect_calls = 0, readback_ns = 0, readbacks = 0, texture_lock_ns = 0;
	uint64_t buffer_ns = 0, texture_ns = 0, heap_ns = 0;
};
inline thread_local FaultStageTime g_fault_stage_time;
[[nodiscard]] uint64_t FaultStageClockNs() noexcept;

// [hib-cp] every 10 s with [hib-flip]: the host memory queries (VirtualQuery) of the borrowed-mode readers,
// on all threads and on the CP (Thread_Gpu), the CP's processors and affinity, and the busiest threads.
void NoteHostQuery(uint64_t ns) noexcept;
// ClampRangeSize in the hybrid: ranges inside one donated mapping, and the rest by the host's region type.
inline std::atomic<uint64_t> g_clamp_donated {0}, g_clamp_image {0}, g_clamp_mapped {0}, g_clamp_private {0}, g_clamp_other {0};
void SetCpThread() noexcept;

[[nodiscard]] bool            Enabled() noexcept;
[[nodiscard]] const Aps5Host* Host() noexcept;
void                          SetHost(const Aps5Host* host) noexcept;

// The executable's event queues (the guest's): Kyty's interrupt (kind 0: count in fflags, context in data,
// one pending event per context) and video out (kind 1) trigger semantics, built in the executable.
int AddEvent(int64_t eq, uint64_t ident, int16_t filter, void* udata, int kind, bool triggered, int64_t data);
int TriggerEvent(int64_t eq, uint64_t ident, int16_t filter, void* trigger_data);
int DeleteEvent(int64_t eq, uint64_t ident, int16_t filter);

// The executable's automated input (APS5_PAD_INPUT_FILE and what drives it) replaces the pad data while
// it runs; nothing changes otherwise.
void PadInjection(void* pad_data, uint64_t bytes);

// At each guest flip: every 10 s a [hib-flip] line (flips, interval p50/p99, draws per frame; the bench reads the
// draws to detect the world).
void NoteFlip();
// Guest flips presented so far (the frame capture ends a frame when it changes).
[[nodiscard]] uint64_t FlipCount() noexcept;

// F9 arms the executable's frame capture instead of the target dump when APS5_KYTY_CAPTURE_TRIGGER is "F9";
// the capture takes the press once.
[[nodiscard]] bool CaptureOnHotkey() noexcept;
void               PressCaptureHotkey() noexcept;
[[nodiscard]] bool TakeCaptureHotkey() noexcept;

// One of the executable's own helpers by name (KytyGpu*_nid_no_patch), or null outside the hybrid.
template <typename Function>
Function Executable(const char* name) noexcept {
	const auto* host = Host();
	return host == nullptr ? nullptr : reinterpret_cast<Function>(host->resolve(name));
}

// In the hybrid, the executable's implementation of a kernel NID instead of Kyty's own `function`.
template <typename Function>
Function Kernel(const char* nid, Function function) noexcept {
	const auto* host = Host();
	return host == nullptr ? function : reinterpret_cast<Function>(host->resolve(nid));
}

} // namespace Hybrid

#endif
