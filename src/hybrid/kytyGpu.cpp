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
#include "graphics/presentation/window.h"
#include "graphics/host_gpu/renderer/targetDump.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shader.h"
#include "hybrid/hybrid.h"
#include "kernel/memory.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"
#include "loader/systemContent.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <exception>
#include <future>
#include <thread>
#include <mutex>
#include <string>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Loader {
void HybridInitWalker();
bool HybridHandleFault(uint64_t vaddr, uint32_t access, uint64_t rip, void* native_context);
} // namespace Loader

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

// The pipeline cache is keyed by the title's TITLE_ID, which pure Kyty reads from param.json when it loads the
// game; in the hybrid the executable loads it, so read it here (the run's directory has app0 next to the exe).
void LoadTitleForPipelineCache() noexcept {
	try {
		const std::filesystem::path param_json = std::filesystem::path("app0") / "sce_sys" / "param.json";
		if (!std::filesystem::exists(param_json)) {
			std::fprintf(stderr, "[kyty-gpu] no %s: the Vulkan pipeline cache stays off\n", param_json.string().c_str());
			return;
		}
		Loader::SystemContentLoadParamSfo(param_json);
		std::string title_id;
		if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) {
			std::fprintf(stderr, "[kyty-gpu] %s has no TITLE_ID: the Vulkan pipeline cache stays off\n", param_json.string().c_str());
			return;
		}
		const auto cache = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
		std::error_code error;
		const auto bytes = std::filesystem::file_size(cache, error);
		std::fprintf(stderr, "[kyty-gpu] Vulkan pipeline cache %s: %s\n", std::filesystem::absolute(cache).string().c_str(),
		             error ? "absent, written once a minute while running" :(std::to_string(bytes) + " bytes").c_str());
	} catch (const std::exception& error) {
		std::fprintf(stderr, "[kyty-gpu] param.json not loaded (%s): the Vulkan pipeline cache stays off\n", error.what());
	}
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
		    // executable owns, on a thread of the DLL that stays in the window loop as the emulator's main
		    // thread does. The executable's startup thread waits until the subsystems are up.
		    std::promise<void> started;
		    auto               ready = started.get_future();
		    std::thread([started = std::move(started)]() mutable {
			    Common::InitializeThreads();
			    static Common::Subsystems subsystems(false);
			    subsystems.Initialize<Config::Lifecycle>();
			    Config::ConfigOptions options {};
			    options.screen_width          = 1920;
			    options.screen_height         = 1080;
			    options.readback_tiled_images = true;
			    // The game asks the executable for its user (libSceUserService stays in the executable): Kyty's
			    // pad maps players by that id.
			    options.user_id = 0x10000000;
			    // The same knobs as pure Kyty's --vblank-frequency and --present-mode, for measuring without vsync.
			    if (const char* hz = std::getenv("KYTY_HYB_VBLANK_HZ"); hz != nullptr && std::atoi(hz) > 0) {
				    options.vblank_frequency = static_cast<uint32_t>(std::atoi(hz));
			    }
			    if (const char* mode = std::getenv("KYTY_HYB_PRESENT_MODE"); mode != nullptr) {
				    const std::string value(mode);
				    if (value == "Fifo") {
					    options.present_mode = Config::PresentMode::Fifo;
				    } else if (value == "Mailbox") {
					    options.present_mode = Config::PresentMode::Mailbox;
				    } else if (value == "Immediate") {
					    options.present_mode = Config::PresentMode::Immediate;
				    }
			    }
			    // Diagnosis only: writes each shader's guest code, SPIR-V and IR to that folder, like pure Kyty's
			    // --shader-log-direction File without the rest of --graphics-debug-dump.
			    if (const char* dump = std::getenv("KYTY_HYB_SHADER_DUMP"); dump != nullptr && dump[0] != '\0') {
				    options.shader_log_direction = Config::LogDirection::File;
				    options.shader_log_folder    = dump;
				    std::fprintf(stderr, "[kyty-gpu] shaders dumped to %s (KYTY_HYB_SHADER_DUMP)\n", dump);
			    }
			    std::fprintf(stderr, "[kyty-gpu] vblank %u Hz, present mode %d (KYTY_HYB_VBLANK_HZ, KYTY_HYB_PRESENT_MODE)\n",
			                 options.vblank_frequency, static_cast<int>(options.present_mode));
			    Config::Load(options);
			    subsystems.Initialize<Log::Lifecycle>();
			    subsystems.Initialize<Libs::LibKernel::PthreadLifecycle>();
			    subsystems.Initialize<Profiler::Lifecycle>();
			    KYTY_PROFILER_THREAD("Thread_KytyMain");
			    subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
			    // The SRT walker asks Kyty before each guest load, as in pure Kyty: the executable passes
			    // only faults in donated memory on, so a walker load must not rely on faulting.
			    Loader::HybridInitWalker();
			    subsystems.Initialize<Libs::Controller::Lifecycle>();
			    LoadTitleForPipelineCache();
			    subsystems.Initialize<Libs::Graphics::Lifecycle>();
			    started.set_value();
			    Libs::Graphics::WindowRun();
			    // The window closed (Alt+F4): the executable owns the process, so its exit path runs (its shutdown,
			    // then TerminateProcess). quick_exit ended in ExitProcess, whose DLL detach ran this DLL's static
			    // destructors against threads already gone, and the process hung.
			    std::fprintf(stderr, "[kyty-gpu] window closed: asking the executable to exit\n");
			    if (const auto request_exit = Hybrid::Executable<void (*)(int)>("LibcRequestExit_nid_postfix")) {
				    request_exit(0);
				    return;
			    }
			    TerminateProcess(GetCurrentProcess(), 0);
		    }).detach();
		    ready.wait();
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

// Called from inside the executable's vectored handler: must not allocate or lock what the faulting thread
// may hold.
extern "C" __declspec(dllexport) bool kyty_gpu_handle_fault(uint64_t address, uint32_t access, uint64_t rip,
                                                            void* context) {
	return Guarded(
	    "kyty_gpu_handle_fault", [&] { return Loader::HybridHandleFault(address, access, rip, context); }, false);
}

// The executable's frame capture: guest flips presented so far, and a wait until the guest GPU is idle.
extern "C" __declspec(dllexport) uint64_t kyty_gpu_flip_count() {
	return Hybrid::FlipCount();
}

// The executable's frame capture: whether F9 was pressed since the last call (APS5_KYTY_CAPTURE_TRIGGER=F9).
extern "C" __declspec(dllexport) bool kyty_gpu_take_capture_hotkey() {
	return Hybrid::TakeCaptureHotkey();
}

extern "C" __declspec(dllexport) void kyty_gpu_wait_idle() {
	Guarded(
	    "kyty_gpu_wait_idle",
	    []() -> bool {
		    Libs::LibKernel::Memory::HybridWaitGpuIdle();
		    return true;
	    },
	    false);
}

// The executable's frame capture starts the render target dump (KYTY_TARGET_DUMP=1) with the next draw, with the
// GPU idle, so the game and its replay dump the first captured frame.
extern "C" __declspec(dllexport) void kyty_gpu_request_target_dump() {
	Libs::Graphics::TargetDump::StartCapture();
}

// The executable's frame capture: Kyty's state from the calls made before it starts (mapped shaders,
// tessellation rings, video out ports) as words, written with the GPU idle and put back by the replay.
constexpr uint64_t CaptureStateMagic = 0x3354535954594bull; // "KYTYST3"

extern "C" __declspec(dllexport) uint64_t kyty_gpu_capture_state(uint64_t* out, uint64_t capacity) {
	return Guarded(
	    "kyty_gpu_capture_state",
	    [&]() -> uint64_t {
		    std::vector<uint64_t> words {CaptureStateMagic, 0};
		    Libs::Graphics::ShaderForEachMapped([&](uint64_t addr, const Libs::Graphics::ShaderMappedData& data) {
			    words[1]++;
			    words.insert(words.end(), {addr, static_cast<uint64_t>(data.type), reinterpret_cast<uint64_t>(data.user_data),
			                               reinterpret_cast<uint64_t>(data.input_semantics), data.num_input_semantics,
			                               (static_cast<uint64_t>(data.code_size_bytes) << 32u) | data.scratch_size_dwords});
		    });
		    uint64_t tessellation[5] {};
		    Libs::Graphics::HybridGetTessellationState(tessellation);
		    words.insert(words.end(), std::begin(tessellation), std::end(tessellation));
		    const auto registers_at = words.size();
		    words.push_back(0);
		    Libs::Graphics::HybridCaptureRegisters(words);
		    words[registers_at] = words.size() - registers_at - 1;
		    const auto blocked_at = words.size();
		    words.push_back(0);
		    Libs::Graphics::HybridCaptureBlocked(words);
		    words[blocked_at] = words.size() - blocked_at - 1;
		    Libs::VideoOut::HybridCaptureState(words);
		    if (out != nullptr && capacity >= words.size()) {
			    std::copy(words.begin(), words.end(), out);
		    }
		    return words.size();
	    },
	    uint64_t {0});
}

extern "C" __declspec(dllexport) bool kyty_gpu_restore_state(const uint64_t* words, uint64_t count) {
	return Guarded(
	    "kyty_gpu_restore_state",
	    [&]() -> bool {
		    if (words == nullptr || count < 7 || words[0] != CaptureStateMagic || count < 2 + words[1] * 6 + 5) {
			    std::fprintf(stderr, "[kyty-gpu] restore: not a Kyty state record (%llu words)\n", static_cast<unsigned long long>(count));
			    return false;
		    }
		    const auto* at = words + 2;
		    for (uint64_t i = 0; i < words[1]; i++, at += 6) {
			    Libs::Graphics::ShaderMappedData data;
			    data.type                = static_cast<Libs::Graphics::Prospero::ShaderBinaryType>(at[1]);
			    data.user_data           = reinterpret_cast<Libs::Graphics::ShaderUserData*>(at[2]);
			    data.input_semantics     = reinterpret_cast<Libs::Graphics::ShaderSemantic*>(at[3]);
			    data.num_input_semantics = static_cast<uint32_t>(at[4]);
			    data.code_size_bytes     = static_cast<uint32_t>(at[5] >> 32u);
			    data.scratch_size_dwords = static_cast<uint32_t>(at[5]);
			    Libs::Graphics::ShaderMapUserData(at[0], data);
		    }
		    std::fprintf(stderr, "[kyty-gpu] restore: %llu shaders mapped\n", static_cast<unsigned long long>(words[1]));
		    Libs::Graphics::HybridSetTessellationState(at);
		    at += 5;
		    std::fputs("[kyty-gpu] restore: tessellation rings set\n", stderr);
		    const auto registers = at < words + count ? *at++ : 0;
		    if (registers > static_cast<uint64_t>(words + count - at) || !Libs::Graphics::HybridRestoreRegisters(at, at + registers)) {
			    std::fputs("[kyty-gpu] restore: the command processor registers do not match this DLL\n", stderr);
			    return false;
		    }
		    at += registers;
		    std::fprintf(stderr, "[kyty-gpu] restore: command processor registers set (%llu words)\n", static_cast<unsigned long long>(registers));
		    const auto blocked = at < words + count ? *at++ : 0;
		    if (blocked > static_cast<uint64_t>(words + count - at) || !Libs::Graphics::HybridRestoreBlocked(at, at + blocked)) {
			    std::fputs("[kyty-gpu] restore: the blocked submissions could not be queued\n", stderr);
			    return false;
		    }
		    at += blocked;
		    std::fprintf(stderr, "[kyty-gpu] restore: blocked submissions queued (%llu words)\n", static_cast<unsigned long long>(blocked));
		    const bool restored = Libs::VideoOut::HybridRestoreState(at, words + count);
		    std::fprintf(stderr, "[kyty-gpu] restored %llu shaders, the tessellation rings and the video out ports: %s\n",
		                 static_cast<unsigned long long>(words[1]), restored ? "ok" : "FAILED");
		    return restored;
	    },
	    false);
}

// The replay's memory is the process's own, not a donated section: the GPU reads it in place.
extern "C" __declspec(dllexport) void kyty_gpu_map_in_place(uint64_t address, uint64_t bytes) {
	Guarded(
	    "kyty_gpu_map_in_place",
	    [&]() -> bool {
		    Libs::LibKernel::Memory::HybridMapPrivate(address, bytes);
		    return true;
	    },
	    false);
}

// The replay wrote guest memory behind Kyty's back (a delta): the GPU caches drop what they hold there.
extern "C" __declspec(dllexport) bool kyty_gpu_invalidate(uint64_t address, uint64_t bytes) {
	return Guarded("kyty_gpu_invalidate", [&] { return Libs::Graphics::HybridInvalidateMemory(address, bytes); }, false);
}

// The frame capture's wait for an idle GPU, bounded: the calling game thread may be the one that writes
// the memory the GPU waits on, so an unbounded wait would freeze the game.
extern "C" __declspec(dllexport) bool kyty_gpu_wait_idle_for(uint32_t millis) {
	return Guarded(
	    "kyty_gpu_wait_idle_for",
	    [&] {
		    if (Libs::LibKernel::Memory::HybridWaitGpuIdleFor(millis)) {
			    return true;
		    }
		    std::fprintf(stderr, "[kyty-gpu] the GPU is still busy after %u ms:\n%s", millis,
		                 Libs::LibKernel::Memory::HybridDescribeGpuBusy().c_str());
		    return false;
	    },
	    false);
}

// Quiet: nothing processed or executed and every queued submission blocked; `blocked` gets their number.
extern "C" __declspec(dllexport) bool kyty_gpu_wait_quiet_for(uint32_t millis, uint32_t* blocked) {
	return Guarded(
	    "kyty_gpu_wait_quiet_for",
	    [&] {
		    if (Libs::LibKernel::Memory::HybridWaitGpuQuietFor(millis, blocked)) {
			    return true;
		    }
		    std::fprintf(stderr, "[kyty-gpu] the GPU is not quiet after %u ms:\n%s", millis,
		                 Libs::LibKernel::Memory::HybridDescribeGpuBusy().c_str());
		    return false;
	    },
	    false);
}

// The replay's offline check that a requeued submission waits on its rewind until the bit is set.
extern "C" __declspec(dllexport) bool kyty_gpu_self_test_rewind() {
	return Guarded("kyty_gpu_self_test_rewind", [] { return Libs::Graphics::HybridSelfTestRewind(); }, false);
}

// The frame capture, with the GPU quiet and before its snapshot: GPU-written images to guest memory.
extern "C" __declspec(dllexport) bool kyty_gpu_flush_images() {
	return Guarded(
	    "kyty_gpu_flush_images",
	    [] {
		    Libs::Graphics::HybridFlushGpuImages();
		    return true;
	    },
	    false);
}

// The same, for the graphics queue only (compute queues may keep running); on timeout it logs the busy queues.
extern "C" __declspec(dllexport) bool kyty_gpu_wait_gfx_idle_for(uint32_t millis) {
	return Guarded(
	    "kyty_gpu_wait_gfx_idle_for",
	    [&] {
		    if (Libs::LibKernel::Memory::HybridWaitGpuGfxIdleFor(millis)) {
			    return true;
		    }
		    std::fprintf(stderr, "[kyty-gpu] the graphics queue is still busy after %u ms:\n%s", millis,
		                 Libs::LibKernel::Memory::HybridDescribeGpuBusy().c_str());
		    return false;
	    },
	    false);
}

// The queues with queued submissions into `out` (0 = gfx); returns how many there are.
extern "C" __declspec(dllexport) uint32_t kyty_gpu_busy_queues(uint32_t* out, uint32_t capacity) {
	return Guarded(
	    "kyty_gpu_busy_queues",
	    [&]() -> uint32_t {
		    const auto busy = Libs::LibKernel::Memory::HybridGpuBusyQueues();
		    for (uint32_t i = 0; out != nullptr && i < capacity && i < busy.size(); i++) {
			    out[i] = busy[i];
		    }
		    return static_cast<uint32_t>(busy.size());
	    },
	    0u);
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
