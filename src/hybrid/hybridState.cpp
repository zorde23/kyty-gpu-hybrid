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

} // namespace Hybrid
