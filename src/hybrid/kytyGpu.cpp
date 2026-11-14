// kyty_gpu.dll: the Kyty GPU libraries (Graphics5 = Agc, Graphics5Driver = AgcDriver, VideoOut) and Pad as a
// DLL inside the AnyPS5 executable, which resolves them by NID and reports its guest memory to them. Every
// export catches at the border: no exception crosses into the executable (hybrid requirement 8).
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/agc.h"
#include "libs/controller.h"
#include "hybrid/hybrid.h"
#include "kernel/memory.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>

namespace Libs {
LIB_DEFINE(InitAgcDriver_1);
LIB_DEFINE(InitPad_1);
namespace LibGen5 {
LIB_DEFINE(InitVideoOut_1);
} // namespace LibGen5
} // namespace Libs

namespace {

Loader::SymbolDatabase& Database() {
	static Loader::SymbolDatabase database;
	static std::once_flag         once;
	std::call_once(once, [] {
		Libs::InitAgcDriver_1(&database);
		Libs::LibGen5::InitVideoOut_1(&database);
		Libs::InitPad_1(&database);
	});
	return database;
}

void Failed(const char* export_name, const char* what) noexcept {
	std::fprintf(stderr, "[kyty-gpu] %s failed: %s\n", export_name, what);
	std::fflush(stderr);
}

template <typename Body>
auto Guarded(const char* export_name, Body&& body, decltype(body()) fallback) noexcept {
	try {
		return body();
	} catch (const std::exception& error) {
		Failed(export_name, error.what());
	} catch (...) {
		Failed(export_name, "unknown exception");
	}
	return fallback;
}

} // namespace

// The SysV-ABI entry point of the Kyty function or object registered under this NID, or 0 when Kyty
// does not have it.
extern "C" __declspec(dllexport) uint64_t kyty_gpu_resolve(const char* nid) {
	return Guarded(
	    "kyty_gpu_resolve",
	    [&]() -> uint64_t {
		    if (nid == nullptr) {
			    return 0;
		    }
		    const std::string key(nid);
		    const auto&       database = Database();
		    for (const auto type: {Loader::SymbolType::Func, Loader::SymbolType::Object}) {
			    if (const auto* record = database.FindByNid(key, type); record != nullptr) {
				    return record->vaddr;
			    }
		    }
		    return 0;
	    },
	    uint64_t {0});
}

extern "C" __declspec(dllexport) void kyty_gpu_init(const Aps5Host* host) {
	Guarded(
	    "kyty_gpu_init",
	    [&]() -> bool {
		    if (host == nullptr || host->version != 2) {
			    Failed("kyty_gpu_init", "unsupported host table");
			    return false;
		    }
		    Hybrid::SetHost(host);
		    // What the emulator's Init runs, minus the loader, network, file system and audio, which the
		    // executable owns. Runs on the executable's startup thread, before any guest code.
		    Common::InitializeThreads();
		    static Common::Subsystems subsystems(false);
		    subsystems.Initialize<Config::Lifecycle>();
		    Config::Load(Config::ConfigOptions {});
		    subsystems.Initialize<Log::Lifecycle>();
		    subsystems.Initialize<Libs::LibKernel::PthreadLifecycle>();
		    subsystems.Initialize<Profiler::Lifecycle>();
		    subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
		    subsystems.Initialize<Libs::Controller::Lifecycle>();
		    subsystems.Initialize<Libs::Graphics::Lifecycle>();
		    return true;
	    },
	    false);
}

extern "C" __declspec(dllexport) void kyty_gpu_on_map(uint64_t address, uint64_t bytes, uint32_t protection,
                                                      void* section, uint64_t offset) {
	Guarded(
	    "kyty_gpu_on_map",
	    [&]() -> bool {
		    Libs::LibKernel::Memory::HybridOnMap(address, bytes, protection, section, offset);
		    return true;
	    },
	    false);
}

extern "C" __declspec(dllexport) void kyty_gpu_on_unmap(uint64_t address, uint64_t bytes) {
	Guarded(
	    "kyty_gpu_on_unmap",
	    [&]() -> bool {
		    Libs::LibKernel::Memory::HybridOnUnmap(address, bytes);
		    return true;
	    },
	    false);
}

extern "C" __declspec(dllexport) void kyty_gpu_on_protect(uint64_t address, uint64_t bytes,
                                                          uint32_t protection) {
	Guarded(
	    "kyty_gpu_on_protect",
	    [&]() -> bool {
		    Libs::LibKernel::Memory::HybridOnProtect(address, bytes, protection);
		    return true;
	    },
	    false);
}

extern "C" __declspec(dllexport) void kyty_gpu_on_section_closed(void* section) {
	Guarded(
	    "kyty_gpu_on_section_closed",
	    [&]() -> bool {
		    Libs::LibKernel::Memory::HybridOnSectionClosed(section);
		    return true;
	    },
	    false);
}

namespace Loader {
bool HybridHandleFault(uint64_t vaddr, uint32_t access, uint64_t rip, void* native_context);
} // namespace Loader

// Called from inside the executable's vectored handler: must not allocate or lock what the faulting thread
// may hold.
extern "C" __declspec(dllexport) bool kyty_gpu_handle_fault(uint64_t address, uint32_t access, uint64_t rip,
                                                            void* context) {
	return Guarded(
	    "kyty_gpu_handle_fault", [&] { return Loader::HybridHandleFault(address, access, rip, context); }, false);
}

extern "C" __declspec(dllexport) void kyty_gpu_prepare_host_write(uint64_t address, uint64_t bytes) {
	Guarded(
	    "kyty_gpu_prepare_host_write",
	    [&]() -> bool {
		    Libs::LibKernel::Memory::HybridPrepareHostWrite(address, bytes);
		    return true;
	    },
	    false);
}
