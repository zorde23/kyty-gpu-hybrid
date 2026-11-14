#include "hybrid/hybrid.h"

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

namespace {
template <typename Function>
Function Executable(const char* name) {
	static_assert(sizeof(Function) == sizeof(uint64_t));
	const auto* host = Host();
	return host == nullptr ? nullptr : reinterpret_cast<Function>(host->resolve(name));
}
} // namespace

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

} // namespace Hybrid
