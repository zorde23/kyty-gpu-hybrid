#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TARGETDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TARGETDUMP_H_

#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics {
class CommandBuffer;
class RenderContext;
struct DrawRenderState;
} // namespace Libs::Graphics

// Render target dump (KYTY_TARGET_DUMP=1): for one guest frame, started with F9 or at guest flip
// KYTY_TARGET_DUMP_AT_FLIP, every draw's color targets are read back after the draw and saved as
// BMP files in _TargetDump/frame_<n>, with index.csv naming each draw's shaders. Slow (the GPU is
// waited for after each draw): only for finding the draw that renders something wrong.
namespace Libs::Graphics::TargetDump {

void Init();
[[nodiscard]] bool Enabled() noexcept;
void RequestCapture() noexcept;
// Starts the dump with the next draw instead of the next flip (the frame capture calls it with the GPU idle).
void StartCapture() noexcept;
// Called at each guest flip (presentation thread).
void OnFlip() noexcept;
// Called by the execution thread after a draw is recorded.
void AfterDraw(RenderContext& context, CommandBuffer& buffer, const DrawRenderState& state,
               vk::Pipeline pipeline);

} // namespace Libs::Graphics::TargetDump

#endif
