#include "common/profiler.h"

#include "common/emulatorConfig.h"
#include "hybrid/hybrid.h"

#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <tracy/Tracy.hpp>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Profiler {

void SetThreadName(const char* name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// The hybrid's crash and fault lines name threads by host id: say which id is which Kyty thread.
	if (Hybrid::Enabled() && name != nullptr) {
		if (std::strcmp(name, "Thread_Gpu") == 0) {
			Hybrid::SetCpThread();
		}
		std::fprintf(stderr, "[hib-thread] %s tid %lu\n", name, ::GetCurrentThreadId());
		using Describe = HRESULT(WINAPI*)(HANDLE, PCWSTR);
		static const auto describe = reinterpret_cast<Describe>(reinterpret_cast<void*>(::GetProcAddress(::GetModuleHandleA("kernel32.dll"), "SetThreadDescription")));
		if (describe != nullptr) {
			const std::wstring wide(name, name + std::strlen(name));
			describe(::GetCurrentThread(), wide.c_str());
		}
	}
#endif
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

void Initialize() {
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
}

void Shutdown() {
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler
