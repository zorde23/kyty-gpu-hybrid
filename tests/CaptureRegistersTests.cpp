#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

using namespace Libs::Graphics;

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		g_failures++;
	}
}

// The handlers these registers reach only write the decoded state, never the renderer.
RenderContext& NoRenderer() {
	alignas(64) static unsigned char storage[64] {};
	return *reinterpret_cast<RenderContext*>(storage);
}

void Packet(CommandProcessor& cp, uint32_t op, uint32_t offset, std::initializer_list<uint32_t> values) {
	std::vector<uint32_t> body {offset};
	body.insert(body.end(), values);
	const auto cmd_id = KYTY_PM4(body.size() + 1u, op, 0);
	const auto size   = static_cast<uint32_t>(body.size());
	if (op == Pm4::IT_SET_CONTEXT_REG) {
		CpOpSetContextReg(cp, cmd_id, body.data(), 0, size);
	} else if (op == Pm4::IT_SET_SH_REG) {
		CpOpSetShaderReg(cp, cmd_id, body.data(), 0, size);
	} else {
		CpOpSetUconfigReg(cp, cmd_id, body.data(), 0, size);
	}
}

template <typename State>
bool Same(const State& a, const State& b) {
	return std::memcmp(&a, &b, sizeof(State)) == 0;
}

bool RoundTrip(CommandProcessor& source, uint64_t expected_registers) {
	std::vector<uint64_t> words;
	source.AppendRegisterWrites(words);
	CommandProcessor target(NoRenderer(), 0);
	target.Reset();
	const uint64_t* at      = words.data();
	uint64_t        applied = 0;
	uint64_t        refused = 0;
	const bool      parsed  = target.ApplyRegisterWrites(at, words.data() + words.size(), applied, refused);
	Check(parsed && at == words.data() + words.size(), "the record parses to its end");
	Check(applied == expected_registers && refused == 0, "every written register is replayed");
	const bool same = Same(source.GetCtx(), target.GetCtx()) && Same(source.GetUcfg(), target.GetUcfg()) &&
	                  Same(source.GetShCtx(), target.GetShCtx());
	Check(same, "the rebuilt context, user config and shader state equal the source");
	std::vector<uint64_t> again;
	target.AppendRegisterWrites(again);
	Check(again == words, "capturing the rebuilt state gives the same record");
	return same;
}

} // namespace

int main() {
	CommandProcessor cp(NoRenderer(), 0);
	cp.Reset();
	Check(RoundTrip(cp, 0), "a reset processor round-trips with no registers");

	Packet(cp, Pm4::IT_SET_CONTEXT_REG, Pm4::PA_SC_SCREEN_SCISSOR_TL, {0x00100020u, 0x04380780u});
	Packet(cp, Pm4::IT_SET_CONTEXT_REG, Pm4::DB_STENCIL_CONTROL, {0x00000200u});
	Packet(cp, Pm4::IT_SET_CONTEXT_REG, Pm4::DB_SHADER_CONTROL, {0x00000013u});
	Packet(cp, Pm4::IT_SET_SH_REG, Pm4::SPI_SHADER_USER_DATA_PS_0, {1u, 2u, 3u, 4u});
	Packet(cp, Pm4::IT_SET_SH_REG, Pm4::COMPUTE_NUM_THREAD_X, {64u, 1u, 1u});
	Packet(cp, Pm4::IT_SET_UCONFIG_REG, Pm4::VGT_PRIMITIVE_TYPE, {4u});
	Check(RoundTrip(cp, 2 + 1 + 1 + 4 + 3 + 1), "written registers round-trip");

	cp.ApplyContextStateOperation(ContextStateOperation::PushClear);
	Packet(cp, Pm4::IT_SET_CONTEXT_REG, Pm4::DB_STENCIL_CONTROL, {0x00000100u});
	Check(RoundTrip(cp, 1 + 4 + 3 + 1), "a cleared and rewritten context round-trips");
	cp.ApplyContextStateOperation(ContextStateOperation::Pop);
	Check(RoundTrip(cp, 2 + 1 + 1 + 4 + 3 + 1), "a popped context round-trips with the saved registers");

	cp.Reset();
	Check(RoundTrip(cp, 0), "a reset after writes round-trips with no registers");

	std::printf("%s\n", g_failures == 0 ? "capture registers: all checks passed" : "capture registers: FAILED");
	return g_failures == 0 ? 0 : 1;
}
