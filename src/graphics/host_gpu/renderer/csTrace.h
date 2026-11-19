#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CSTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CSTRACE_H_

#include <cstdint>
#include <span>
#include <string>
#include <utility>

// Diagnosis only (KYTY_HYB_TRACE_CS=<shader hash in hex>): logs each dispatch of that compute shader
// with its buffers and the first words the guest sees in them, every write the command stream makes
// to one of those buffers between dispatches, and every DMA the 0x3022c rule drops.
namespace Libs::Graphics::CsTrace {

bool Enabled();
// Any dispatch goes to a ring of recent events; the first one of the traced shader also lists the
// events that touched its buffers. True when the caller should read its first buffer from the GPU
// (KYTY_HYB_TRACE_CS_SYNC=1) and pass it to NoteGpuWords.
bool NoteDispatch(uint64_t shader_hash, std::span<const std::pair<uint64_t, uint64_t>> buffers,
                  uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
void NoteGpuWords(uint64_t address, std::span<const uint32_t> words);
bool SyncRequested();
// The buffers of the first MaxDumpedDispatches dispatches are written with KYTY_HYB_TRACE_CS_DUMP.
inline constexpr uint64_t MaxDumpedDispatches = 32;
uint64_t LastDispatchIndex();
// KYTY_HYB_TRACE_CS_DUMP: where those buffers are written, empty when not asked.
const std::string& DumpFolder();
// The push data and the host descriptors of a dispatch of the traced shader, for a replay outside the game.
void NoteBindings(uint64_t shader_hash, std::span<const uint32_t> shader_data,
                  std::span<const std::pair<uint64_t, uint64_t>> descriptors);
void NoteWrite(const char* what, uint64_t address, uint64_t size, uint64_t value);
void NoteDroppedDma(uint64_t address, uint64_t size, uint8_t src_sel, uint64_t source);

} // namespace Libs::Graphics::CsTrace

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CSTRACE_H_
