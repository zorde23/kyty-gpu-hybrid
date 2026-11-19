#include "graphics/host_gpu/renderer/csTrace.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::CsTrace {

namespace {

constexpr uint32_t MaxLines = 20000;
constexpr size_t   RingSize = 8192;

// A write of the command stream (hash 0) or a dispatch of any shader (ranges are its buffers).
struct Event {
	uint64_t                                   sequence = 0;
	const char*                                what     = "";
	uint64_t                                   hash     = 0;
	uint64_t                                   address  = 0;
	uint64_t                                   size     = 0;
	uint64_t                                   value    = 0;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
};

struct State {
	bool                                       enabled = false;
	bool                                       sync    = false;
	std::string                                dump_folder;
	uint64_t                                   hash    = 0;
	std::mutex                                 mutex;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	uint64_t                                   dispatches = 0;
	std::atomic<uint32_t>                      lines {0};
	std::vector<Event>                         ring = std::vector<Event>(RingSize);
	uint64_t                                   events = 0;
};

bool Overlaps(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) { return a < b + b_size && b < a + a_size; }

State& Get() {
	static State state;
	static const bool parsed = [] {
		if (const char* value = std::getenv("KYTY_HYB_TRACE_CS"); value != nullptr && value[0] != '\0') {
			state.hash    = std::strtoull(value, nullptr, 16);
			state.enabled = state.hash != 0;
			if (state.enabled) {
				const char* sync = std::getenv("KYTY_HYB_TRACE_CS_SYNC");
				state.sync       = sync != nullptr && sync[0] == '1';
				if (const char* dump = std::getenv("KYTY_HYB_TRACE_CS_DUMP"); dump != nullptr && dump[0] != '\0') {
					state.dump_folder = dump;
					std::fprintf(stderr, "[cs-trace] the buffers of its first %" PRIu64 " dispatches go to %s (KYTY_HYB_TRACE_CS_DUMP)\n",
					             MaxDumpedDispatches, dump);
				}
				std::fprintf(stderr, "[cs-trace] tracing cs %016" PRIx64 " (KYTY_HYB_TRACE_CS)%s\n", state.hash,
				             state.sync ? ", GPU read of its first buffer at the first dispatch (KYTY_HYB_TRACE_CS_SYNC)" : "");
			} else {
				std::fprintf(stderr, "[cs-trace] KYTY_HYB_TRACE_CS=%s is not a shader hash: tracing off\n", value);
			}
		}
		return true;
	}();
	(void)parsed;
	return state;
}

bool TakeLine() { return Get().lines.fetch_add(1, std::memory_order_relaxed) < MaxLines; }

// The guest's view of the first words: a plain read of mapped memory, no synchronization with the GPU.
std::string GuestWords(uint64_t address, uint64_t size) {
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0 ||
	    info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
		return "unmapped";
	}
	const auto available = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize - address;
	const auto count     = static_cast<uint32_t>(std::min<uint64_t>({16, size / 4, available / 4}));
	std::string text;
	for (uint32_t i = 0; i < count; i++) {
		char word[16];
		std::snprintf(word, sizeof(word), "%s%08" PRIx32, i == 0 ? "" : " ",
		              reinterpret_cast<const volatile uint32_t*>(address)[i]);
		text += word;
	}
	return text;
}

} // namespace

bool Enabled() { return Get().enabled; }

bool SyncRequested() { return Get().sync; }

const std::string& DumpFolder() { return Get().dump_folder; }

uint64_t LastDispatchIndex() {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	return state.dispatches - 1;
}

bool NoteDispatch(uint64_t shader_hash, std::span<const std::pair<uint64_t, uint64_t>> buffers,
                  uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
	auto& state = Get();
	if (!state.enabled) {
		return false;
	}
	std::lock_guard lock(state.mutex);
	if (shader_hash != state.hash) {
		auto& event    = state.ring[state.events % RingSize];
		event.sequence = state.events++;
		event.what     = "dispatch";
		event.hash     = shader_hash;
		event.ranges.assign(buffers.begin(), buffers.end());
		return false;
	}
	const auto index = state.dispatches++;
	if (index == 0) {
		// What touched these buffers before the first dispatch: the oldest first.
		const auto first = state.events > RingSize ? state.events - RingSize : 0;
		for (auto sequence = first; sequence < state.events; sequence++) {
			const auto& event = state.ring[sequence % RingSize];
			for (size_t i = 0; i < buffers.size(); i++) {
				const auto& [base, length] = buffers[i];
				bool hit = event.hash == 0 && Overlaps(event.address, event.size, base, length);
				for (const auto& [address, size]: event.ranges) {
					hit = hit || (event.hash != 0 && Overlaps(address, size, base, length));
				}
				if (!hit || !TakeLine()) {
					continue;
				}
				if (event.hash == 0) {
					std::fprintf(stderr,
					             "[cs-trace] before: event %" PRIu64 " %s 0x%012" PRIx64 " size 0x%" PRIx64 " value 0x%" PRIx64
					             " touches buffer %zu\n", sequence, event.what, event.address, event.size, event.value, i);
				} else {
					std::fprintf(stderr, "[cs-trace] before: event %" PRIu64 " dispatch of cs %016" PRIx64 " binds buffer %zu\n",
					             sequence, event.hash, i);
				}
			}
		}
		std::fprintf(stderr, "[cs-trace] %" PRIu64 " events seen before the first dispatch\n", state.events);
	}
	for (const auto& range: buffers) {
		if (range.first != 0 && std::find(state.ranges.begin(), state.ranges.end(), range) == state.ranges.end()) {
			state.ranges.push_back(range);
		}
	}
	if (!TakeLine()) {
		return false;
	}
	std::fprintf(stderr, "[cs-trace] dispatch %" PRIu64 " groups %ux%ux%u, %zu buffers\n", index, groups_x, groups_y,
	             groups_z, buffers.size());
	for (size_t i = 0; i < buffers.size(); i++) {
		std::fprintf(stderr, "[cs-trace]   buffer %zu 0x%012" PRIx64 " size 0x%" PRIx64 " guest %s\n", i, buffers[i].first,
		             buffers[i].second, GuestWords(buffers[i].first, buffers[i].second).c_str());
	}
	return (index == 0 && state.sync) || (index < MaxDumpedDispatches && !state.dump_folder.empty());
}

void NoteGpuWords(uint64_t address, std::span<const uint32_t> words) {
	std::string text;
	for (const auto word: words) {
		char item[16];
		std::snprintf(item, sizeof(item), "%s%08" PRIx32, text.empty() ? "" : " ", word);
		text += item;
	}
	std::fprintf(stderr, "[cs-trace] GPU copy of 0x%012" PRIx64 " at the first dispatch: %s\n", address, text.c_str());
}

void NoteBindings(uint64_t shader_hash, std::span<const uint32_t> shader_data,
                  std::span<const std::pair<uint64_t, uint64_t>> descriptors) {
	auto& state = Get();
	if (!state.enabled || shader_hash != state.hash || !TakeLine()) {
		return;
	}
	std::string text;
	for (const auto word: shader_data) {
		char item[16];
		std::snprintf(item, sizeof(item), " %08" PRIx32, word);
		text += item;
	}
	std::fprintf(stderr, "[cs-trace]   push data of dispatch %" PRIu64 " (%zu words):%s\n", state.dispatches - 1,
	             shader_data.size(), text.c_str());
	for (size_t i = 0; i < descriptors.size(); i++) {
		std::fprintf(stderr, "[cs-trace]   descriptor %zu offset 0x%" PRIx64 " range 0x%" PRIx64 "\n", i, descriptors[i].first,
		             descriptors[i].second);
	}
}

void NoteWrite(const char* what, uint64_t address, uint64_t size, uint64_t value) {
	auto& state = Get();
	if (!state.enabled) {
		return;
	}
	std::lock_guard lock(state.mutex);
	auto&           event = state.ring[state.events % RingSize];
	event.sequence        = state.events++;
	event.what            = what;
	event.hash            = 0;
	event.address         = address;
	event.size            = size;
	event.value           = value;
	event.ranges.clear();
	for (const auto& [base, length]: state.ranges) {
		if (address < base + length && base < address + size) {
			if (TakeLine()) {
				std::fprintf(stderr,
				             "[cs-trace] %s 0x%012" PRIx64 " size 0x%" PRIx64 " value 0x%" PRIx64
				             " (buffer 0x%012" PRIx64 ", after dispatch %" PRIu64 ")\n",
				             what, address, size, value, base, state.dispatches);
			}
			return;
		}
	}
}

void NoteDroppedDma(uint64_t address, uint64_t size, uint8_t src_sel, uint64_t source) {
	if (!Get().enabled || !TakeLine()) {
		return;
	}
	std::fprintf(stderr, "[cs-trace] dma dropped by the 0x3022c rule: dst 0x%012" PRIx64 " size 0x%" PRIx64
	             " src_sel %u src 0x%" PRIx64 "\n", address, size, src_sel, source);
}

} // namespace Libs::Graphics::CsTrace
