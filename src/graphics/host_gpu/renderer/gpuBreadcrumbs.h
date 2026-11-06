#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {
struct GraphicContext;
}

// GPU hang breadcrumbs (KYTY_GPU_BREADCRUMBS=1 and VK_AMD_buffer_marker): every draw and dispatch
// writes its id to a host-visible buffer when it starts (top of pipe) and when it ends (bottom of
// pipe). After a device loss the fatal error report names the draws that started and never ended,
// with the shader hashes of their pipelines.
namespace Libs::Graphics::GpuBreadcrumbs {

void Init(GraphicContext& graphics);
[[nodiscard]] bool Enabled() noexcept;

// Shader hashes of a pipeline, for the report (any thread).
void RegisterPipeline(vk::Pipeline pipeline, const char* kind, uint64_t h0, uint64_t h1 = 0,
                      uint64_t h2 = 0, uint64_t h3 = 0);
// The pipeline the recording thread bound last, per bind point.
void NoteBind(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) noexcept;
// A new id for a draw (graphics) or dispatch (compute), remembered with the bound pipeline.
[[nodiscard]] uint32_t Next(bool compute, const char* what) noexcept;
// Writes the id at the start (begin) or end of the commands before it.
void Write(vk::CommandBuffer c, uint32_t id, bool begin);

} // namespace Libs::Graphics::GpuBreadcrumbs

#endif
