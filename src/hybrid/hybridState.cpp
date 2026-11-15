#include "hybrid/hybrid.h"

#include "common/debugCounters.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <array>
#include <string>
#include <mutex>
#include <vector>

#include <atomic>
#include <unordered_map>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>
#endif

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

namespace {
std::atomic<uint64_t> g_query_calls {0}, g_query_ns {0}, g_cp_query_calls {0}, g_cp_query_ns {0}, g_cp_processors {0};
thread_local bool     t_is_cp = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
std::atomic<HANDLE> g_cp_thread {nullptr};

// Sampled (1 of 16) VirtualQuery callers: three frames above the query helper, and the calling thread.
struct Tallies {
	std::mutex                                            mutex;
	std::vector<std::pair<std::array<uint64_t, 3>, std::pair<uint64_t, uint64_t>>> rows;
};
Tallies g_query_sites, g_query_threads;

void Tally(Tallies& tallies, const std::array<uint64_t, 3>& key, uint64_t ns) {
	std::scoped_lock lock(tallies.mutex);
	for (auto& [k, value]: tallies.rows) {
		if (k == key) {
			value.first++;
			value.second += ns;
			return;
		}
	}
	if (tallies.rows.size() < 256) {
		tallies.rows.push_back({key, {1, ns}});
	}
}

std::string Symbol(uint64_t address) {
	if (address == 0) {
		return "-";
	}
	HMODULE module = nullptr;
	char    path[MAX_PATH] = "?";
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) && module != nullptr) {
		GetModuleFileNameA(module, path, MAX_PATH);
	}
	const char* file = std::strrchr(path, '\\');
	char        text[MAX_PATH + 32];
	std::snprintf(text, sizeof(text), "%s+0x%llx", file != nullptr ? file + 1 : path, static_cast<unsigned long long>(address - reinterpret_cast<uint64_t>(module)));
	return text;
}

void ReportCallers() {
	for (auto* tallies: {&g_query_sites, &g_query_threads}) {
		std::vector<std::pair<std::array<uint64_t, 3>, std::pair<uint64_t, uint64_t>>> rows;
		{
			std::scoped_lock lock(tallies->mutex);
			rows.swap(tallies->rows);
		}
		std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
		std::string line = tallies == &g_query_sites ? "[hib-cp] VirtualQuery callers (sampled 1/16; calls, ms, 3 frames):" : "[hib-cp] VirtualQuery threads (sampled 1/16; tid, calls, ms):";
		for (size_t i = 0; i < rows.size() && i < 8; ++i) {
			char item[64];
			std::snprintf(item, sizeof(item), " %llu %.2f", static_cast<unsigned long long>(rows[i].second.first), static_cast<double>(rows[i].second.second) / 1e6);
			if (tallies == &g_query_sites) {
				line += item + std::string(" ") + Symbol(rows[i].first[0]) + " < " + Symbol(rows[i].first[1]) + " < " + Symbol(rows[i].first[2]) + ";";
			} else {
				line += " " + std::to_string(rows[i].first[0]) + item + ";";
			}
		}
		std::fprintf(stderr, "%s\n", line.c_str());
	}
}

void ReportThreads(uint64_t frames) {
	static std::unordered_map<DWORD, uint64_t> previous;
	struct Busy {
		DWORD    tid;
		uint64_t ns;
		int      priority;
		std::string name;
		std::string start;
	};
	std::vector<Busy>                    busy;
	std::unordered_map<DWORD, uint64_t> now;
	const auto                           snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return;
	}
	using Describe = HRESULT(WINAPI*)(HANDLE, PWSTR*);
	using Query    = LONG(WINAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
	static const auto describe = reinterpret_cast<Describe>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
	static const auto query    = reinterpret_cast<Query>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryInformationThread")));
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != GetCurrentProcessId()) {
			continue;
		}
		const auto thread = OpenThread(THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) {
			continue;
		}
		FILETIME created {}, exited {}, kernel {}, user {};
		if (GetThreadTimes(thread, &created, &exited, &kernel, &user)) {
			const auto total = ((static_cast<uint64_t>(kernel.dwHighDateTime) << 32u | kernel.dwLowDateTime) +
			                    (static_cast<uint64_t>(user.dwHighDateTime) << 32u | user.dwLowDateTime)) * 100;
			now[entry.th32ThreadID] = total;
			const auto before = previous.find(entry.th32ThreadID);
			Busy b {entry.th32ThreadID, total - (before != previous.end() ? before->second : 0), GetThreadPriority(thread), {}, {}};
			PWSTR text = nullptr;
			if (describe != nullptr && SUCCEEDED(describe(thread, &text)) && text != nullptr) {
				for (const wchar_t* c = text; *c != 0; ++c) {
					b.name += static_cast<char>(*c < 128 ? *c : '?');
				}
				LocalFree(text);
			}
			void* start = nullptr;
			if (query != nullptr && query(thread, 9, &start, sizeof(start), nullptr) == 0 && start != nullptr) {
				HMODULE module = nullptr;
				char    path[MAX_PATH] = "?";
				if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCSTR>(start), &module) && module != nullptr) {
					GetModuleFileNameA(module, path, MAX_PATH);
				}
				const char* file = std::strrchr(path, '\\');
				char        site[MAX_PATH + 32];
				std::snprintf(site, sizeof(site), "%s+0x%llx", file != nullptr ? file + 1 : path, static_cast<unsigned long long>(reinterpret_cast<uint64_t>(start) - reinterpret_cast<uint64_t>(module)));
				b.start = site;
			}
			busy.push_back(std::move(b));
		}
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	previous = std::move(now);
	std::sort(busy.begin(), busy.end(), [](const Busy& a, const Busy& b) { return a.ns > b.ns; });
	std::string line = "[hib-cp] busiest threads (ms per frame, priority, name, start):";
	for (size_t i = 0; i < busy.size() && i < 8; ++i) {
		char item[512];
		std::snprintf(item, sizeof(item), " %lu %.2f p%d %s %s;", busy[i].tid, frames != 0 ? static_cast<double>(busy[i].ns) / 1e6 / static_cast<double>(frames) : 0.0, busy[i].priority, busy[i].name.empty() ? "-" : busy[i].name.c_str(), busy[i].start.empty() ? "?" : busy[i].start.c_str());
		line += item;
	}
	std::fprintf(stderr, "%s\n", line.c_str());
}
#endif

void ReportHostQueries(uint64_t frames) {
	const auto per = [frames](uint64_t value) { return frames != 0 ? static_cast<double>(value) / static_cast<double>(frames) : 0.0; };
	const auto calls = g_query_calls.exchange(0), ns = g_query_ns.exchange(0), cp_calls = g_cp_query_calls.exchange(0), cp_ns = g_cp_query_ns.exchange(0);
	const auto processors = g_cp_processors.exchange(0);
	unsigned long long affinity = 0, ideal = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (const auto cp = g_cp_thread.load()) {
		GROUP_AFFINITY group {};
		PROCESSOR_NUMBER number {};
		if (GetThreadGroupAffinity(cp, &group)) {
			affinity = group.Mask;
		}
		if (GetThreadIdealProcessorEx(cp, &number)) {
			ideal = number.Number;
		}
	}
#endif
	std::fprintf(stderr,
	             "[hib-cp] per frame: VirtualQuery CP %.1f calls %.3f ms, all threads %.1f calls %.3f ms; CP ran on processors 0x%llx, affinity 0x%llx, ideal %llu\n",
	             per(cp_calls), per(cp_ns) / 1e6, per(calls), per(ns) / 1e6, static_cast<unsigned long long>(processors), affinity, ideal);
	const auto donated = g_clamp_donated.exchange(0), image = g_clamp_image.exchange(0), mapped = g_clamp_mapped.exchange(0), priv = g_clamp_private.exchange(0), other = g_clamp_other.exchange(0);
	std::fprintf(stderr, "[hib-cp] ClampRangeSize per frame: inside one donated mapping %.1f; to the system: image %.1f, mapped %.1f, private %.1f, other %.1f\n",
	             per(donated), per(image), per(mapped), per(priv), per(other));
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	ReportThreads(frames);
	ReportCallers();
#endif
}
} // namespace

void NoteHostQuery(uint64_t ns) noexcept {
	const auto call = g_query_calls.fetch_add(1, std::memory_order_relaxed);
	g_query_ns.fetch_add(ns, std::memory_order_relaxed);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (call % 16 == 0) {
		void* frames[4] {};
		const auto captured = RtlCaptureStackBackTrace(1, 4, frames, nullptr);
		std::array<uint64_t, 3> site {};
		for (USHORT i = 1; i < captured && i < 4; ++i) {
			site[i - 1] = reinterpret_cast<uint64_t>(frames[i]);
		}
		Tally(g_query_sites, site, ns);
		Tally(g_query_threads, std::array<uint64_t, 3> {GetCurrentThreadId(), 0, 0}, ns);
	}
#else
	static_cast<void>(call);
#endif
	if (t_is_cp) {
		g_cp_query_calls.fetch_add(1, std::memory_order_relaxed);
		g_cp_query_ns.fetch_add(ns, std::memory_order_relaxed);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		g_cp_processors.fetch_or(uint64_t {1} << (GetCurrentProcessorNumber() & 63u), std::memory_order_relaxed);
#endif
	}
}

void SetCpThread() noexcept {
	t_is_cp = true;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE self = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self, THREAD_QUERY_INFORMATION, FALSE, 0)) {
		g_cp_thread.store(self);
	}
#endif
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
	ReportHostQueries(flips);
	intervals.clear();
	period       = now;
	draws_before = draws;
}

} // namespace Hybrid
