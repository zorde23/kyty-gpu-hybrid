#include "hybrid/hybrid.h"

#include "common/debugCounters.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>

#include <atomic>

namespace Hybrid {

namespace {
std::atomic<const Aps5Host*> g_host {nullptr};
} // namespace

bool Enabled() noexcept {
	return g_host.load(std::memory_order_acquire) != nullptr;
}

const Aps5Host* Host() noexcept {
	return g_host.load(std::memory_order_acquire);
}

void SetHost(const Aps5Host* host) noexcept {
	g_host.store(host, std::memory_order_release);
}

// EBADF, as the executable answers for a queue it does not know.
constexpr int BadQueue = static_cast<int>(0x80020009u);

int AddEvent(int64_t eq, uint64_t ident, int16_t filter, void* udata, int kind, bool triggered, int64_t data) {
	static const auto add = Executable<int (*)(int64_t, uint64_t, int16_t, void*, int, int, int64_t)>(
	    "KytyGpuAddEvent_nid_no_patch");
	return add == nullptr ? BadQueue : add(eq, ident, filter, udata, kind, triggered ? 1 : 0, data);
}

int TriggerEvent(int64_t eq, uint64_t ident, int16_t filter, void* trigger_data) {
	static const auto trigger = Executable<int (*)(int64_t, uint64_t, int16_t, void*)>("KytyGpuTriggerEvent_nid_no_patch");
	return trigger == nullptr ? BadQueue : trigger(eq, ident, filter, trigger_data);
}

int DeleteEvent(int64_t eq, uint64_t ident, int16_t filter) {
	static const auto remove = Executable<int (*)(int64_t, uint64_t, int16_t)>("KytyGpuDeleteEvent_nid_no_patch");
	return remove == nullptr ? BadQueue : remove(eq, ident, filter);
}

void PadInjection(void* pad_data, uint64_t bytes) {
	static const auto inject = Executable<bool (*)(int, void*, uint64_t)>("KytyGpuPadInjection_nid_no_patch");
	if (inject != nullptr) {
		inject(0, pad_data, bytes);
	}
}

uint64_t FaultStageClockNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void NoteFlip() {
	if (!Enabled()) {
		return;
	}
	using Clock = std::chrono::steady_clock;
	static std::mutex              mutex;
	static Clock::time_point       last {};
	static Clock::time_point       period {};
	static std::vector<double>     intervals;
	static uint64_t                draws_before = 0;
	std::lock_guard                lock(mutex);
	const auto                     now = Clock::now();
	if (period == Clock::time_point {}) {
		Common::DebugCounters::SetEnabled(true); // draws/quadro needs the counter even without the overlay
		period       = now;
		draws_before = Common::DebugCounters::Get(Common::DebugCounters::Counter::Draws);
	}
	if (last != Clock::time_point {}) {
		intervals.push_back(std::chrono::duration<double, std::milli>(now - last).count());
	}
	last = now;
	if (now - period < std::chrono::seconds(10) || intervals.empty()) {
		return;
	}
	std::sort(intervals.begin(), intervals.end());
	const auto draws = Common::DebugCounters::Get(Common::DebugCounters::Counter::Draws);
	const auto flips = intervals.size();
	std::fprintf(stderr, "[hib-flip] 10 s: %zu flips, intervalo p50 %.1f ms p99 %.1f ms, draws/quadro %llu\n", flips,
	             intervals[flips / 2], intervals[std::min(flips - 1, flips * 99 / 100)],
	             static_cast<unsigned long long>((draws - draws_before) / flips));
	intervals.clear();
	period       = now;
	draws_before = draws;
}

} // namespace Hybrid
