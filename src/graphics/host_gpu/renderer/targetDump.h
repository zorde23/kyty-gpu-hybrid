#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TARGETDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TARGETDUMP_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {
class CommandBuffer;
class RenderContext;
struct DrawRenderState;
} // namespace Libs::Graphics

// Render target dump (KYTY_TARGET_DUMP=1): for one guest frame, started with F9, when the file
// KYTY_TARGET_DUMP_TRIGGER appears (it is deleted) or at guest flip KYTY_TARGET_DUMP_AT_FLIP, the
// color and depth targets of every draw and the storage images of every dispatch are read back
// and saved as BMP files in _TargetDump/frame_<n>, with index.csv naming each operation's shaders.
// The frame runs from one flip in the command stream to the next. Slow (the GPU is waited for
// after each operation): only for finding the draw that renders something wrong.
namespace Libs::Graphics::TargetDump {

void Init();
[[nodiscard]] bool Enabled() noexcept;
void RequestCapture() noexcept;
// Starts the dump with the next draw instead of the next flip (the frame capture calls it with the GPU idle).
void StartCapture() noexcept;
// While a dump is requested or running, no draw is skipped while its pipeline compiles in the
// background: the draw waits (the replay and the game's reference then render the same draws).
[[nodiscard]] bool NoPipelineSkip() noexcept;
void NoteWaitedPipeline() noexcept;
// Called at each guest flip (presentation thread).
void OnFlip() noexcept;
// Called by the execution thread at each flip in the command stream: the frame boundary.
void OnStreamFlip() noexcept;
// Called by the execution thread after a draw is recorded.
// An image a dispatch writes (or a draw samples), at the view it binds.
struct StorageTarget {
	uint32_t id   = 0; // ImageId index
	uint32_t mip  = 0;
	uint32_t layer = 0;
};
// textures: the pixel shader's sampled images, written as t files for the draws dumped one by one.
void AfterDraw(RenderContext& context, CommandBuffer& buffer, const DrawRenderState& state,
               vk::Pipeline pipeline, std::span<const StorageTarget> textures = {});
// Called by the execution thread after a dispatch is recorded.
void AfterDispatch(RenderContext& context, CommandBuffer& buffer,
                   std::span<const StorageTarget> targets, vk::Pipeline pipeline);

} // namespace Libs::Graphics::TargetDump

#endif
