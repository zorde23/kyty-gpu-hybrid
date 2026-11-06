#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics::GpuBreadcrumbs {

namespace {

struct PipelineInfo {
	const char*             kind = "";
	std::array<uint64_t, 4> hashes {};
};

struct Record {
	std::atomic<uint32_t> id {0};
	const char*           what     = "";
	VkPipeline            pipeline = VK_NULL_HANDLE;
};

constexpr uint32_t RingSize = 1u << 16;

bool                    g_enabled = false;
vk::Buffer              g_buffer  = nullptr;
vk::DeviceMemory        g_memory  = nullptr;
volatile uint32_t*      g_markers = nullptr; // [0] last started id, [1] last ended id
std::atomic<uint32_t>   g_next {0};
std::array<Record, RingSize> g_ring;
thread_local VkPipeline t_graphics = VK_NULL_HANDLE;
thread_local VkPipeline t_compute  = VK_NULL_HANDLE;

std::mutex                                       g_pipelines_mutex;
std::unordered_map<VkPipeline, PipelineInfo>     g_pipelines;

void Report() {
	const uint32_t started = g_markers[0];
	const uint32_t ended   = g_markers[1];
	const uint32_t issued  = g_next.load();
	Log::WriteToConsoleAndLog(fmt::format(
	    "--- GPU breadcrumbs ---\nissued {} draws/dispatches, last started {}, last ended {}\n",
	    issued, started, ended));
	// Commands start in order: the hung ones started after the last one that ended.
	uint32_t first = ended + 1;
	if (started - first > 31) {
		first = started - 31;
	}
	std::lock_guard lock(g_pipelines_mutex);
	for (uint32_t id = first; id <= started + 2 && id <= issued; ++id) {
		const auto& record = g_ring[id % RingSize];
		if (record.id.load() != id) {
			continue;
		}
		std::string line = fmt::format("{} #{} {} pipeline {:#x}", id <= started ? "RUNNING" : "next", id,
		                               record.what, reinterpret_cast<uint64_t>(record.pipeline));
		if (const auto it = g_pipelines.find(record.pipeline); it != g_pipelines.end()) {
			line += fmt::format(" {} {:016x} {:016x} {:016x} {:016x}", it->second.kind,
			                    it->second.hashes[0], it->second.hashes[1], it->second.hashes[2],
			                    it->second.hashes[3]);
		}
		Log::WriteToConsoleAndLog(line + "\n");
	}
}

} // namespace

void Init(GraphicContext& graphics) {
	if (!graphics.buffer_marker_amd_enabled || g_enabled) {
		return;
	}
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = 256;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	EXIT_IF(graphics.device.createBuffer(&buffer_info, nullptr, &g_buffer) != vk::Result::eSuccess);
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(g_buffer, &requirements);
	const auto& props = graphics.physical_device_memory_properties;
	const auto wanted = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
		if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
		    (props.memoryTypes[i].propertyFlags & wanted) == wanted) {
			type = i;
			break;
		}
	}
	EXIT_IF(type == UINT32_MAX);
	vk::MemoryAllocateInfo alloc {};
	alloc.allocationSize  = requirements.size;
	alloc.memoryTypeIndex = type;
	EXIT_IF(graphics.device.allocateMemory(&alloc, nullptr, &g_memory) != vk::Result::eSuccess);
	EXIT_IF(graphics.device.bindBufferMemory(g_buffer, g_memory, 0) != vk::Result::eSuccess);
	void* mapped = nullptr;
	EXIT_IF(graphics.device.mapMemory(g_memory, 0, VK_WHOLE_SIZE, {}, &mapped) != vk::Result::eSuccess);
	g_markers    = static_cast<volatile uint32_t*>(mapped);
	g_markers[0] = 0;
	g_markers[1] = 0;
	g_enabled    = true;
	Common::g_fatal_report_hook = &Report;
	Log::WriteToConsoleAndLog("GPU breadcrumbs: on (VK_AMD_buffer_marker)\n");
}

bool Enabled() noexcept {
	return g_enabled;
}

void RegisterPipeline(vk::Pipeline pipeline, const char* kind, uint64_t h0, uint64_t h1,
                      uint64_t h2, uint64_t h3) {
	if (!g_enabled) {
		return;
	}
	std::lock_guard lock(g_pipelines_mutex);
	g_pipelines[static_cast<VkPipeline>(pipeline)] = {kind, {h0, h1, h2, h3}};
}

void NoteBind(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) noexcept {
	if (bind_point == vk::PipelineBindPoint::eCompute) {
		t_compute = pipeline;
	} else if (bind_point == vk::PipelineBindPoint::eGraphics) {
		t_graphics = pipeline;
	}
}

uint32_t Next(bool compute, const char* what) noexcept {
	const uint32_t id     = g_next.fetch_add(1) + 1;
	auto&          record = g_ring[id % RingSize];
	record.what           = what;
	record.pipeline       = compute ? t_compute : t_graphics;
	record.id.store(id);
	return id;
}

void Write(vk::CommandBuffer c, uint32_t id, bool begin) {
	c.writeBufferMarkerAMD(begin ? vk::PipelineStageFlagBits::eTopOfPipe
	                             : vk::PipelineStageFlagBits::eBottomOfPipe,
	                       g_buffer, begin ? 0 : 4, id);
}

} // namespace Libs::Graphics::GpuBreadcrumbs
