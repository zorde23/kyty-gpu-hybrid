// kyty_gpu.dll (hybrid step 0): the Kyty GPU libraries (Graphics5 = Agc, Graphics5Driver = AgcDriver,
// VideoOut) and Pad (Kyty owns the window and its input) as a DLL that the AnyPS5 executable resolves by NID. Step 0 only resolves; nothing here
// initializes the GPU, kernel or memory, so no export may be called yet.
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

#include <cstdint>
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

} // namespace

// The SysV-ABI entry point of the Kyty function or object registered under this NID, or 0 when Kyty
// does not have it.
extern "C" __declspec(dllexport) uint64_t kyty_gpu_resolve(const char* nid) {
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
}
