#include "graphics/host_gpu/renderer/targetDump.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace Libs::Graphics::TargetDump {

namespace {

enum class State : uint32_t { Idle, Requested, Capturing };

bool                  g_enabled  = false;
uint32_t              g_at_flip  = 0;
uint32_t              g_max      = 3500; // a heavy title frame has ~3200 draws
uint32_t              g_each_from = 1;   // draws dumped one by one (KYTY_TARGET_DUMP_DRAWS=a-b)
uint32_t              g_each_to   = 0;
std::atomic<State>    g_state {State::Idle};
std::atomic<uint32_t> g_flips {0};
std::atomic<uint32_t> g_waited {0};
uint32_t              g_frame    = 0;
uint32_t              g_draw     = 0;
std::FILE*            g_index    = nullptr;
std::filesystem::path g_folder;

vk::Buffer       g_staging = nullptr;
vk::DeviceMemory g_memory  = nullptr;
void*            g_mapped  = nullptr;
vk::DeviceSize   g_size    = 0;

float HalfToFloat(uint16_t h) {
	const uint32_t sign = (h >> 15u) & 1u;
	const uint32_t exp  = (h >> 10u) & 0x1fu;
	const uint32_t man  = h & 0x3ffu;
	float          v    = 0.0f;
	if (exp == 0) {
		v = std::ldexp(static_cast<float>(man), -24);
	} else if (exp == 31) {
		v = man != 0 ? 0.0f : 65504.0f;
	} else {
		v = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
	}
	return sign != 0 ? -v : v;
}

// 11/10-bit unsigned floats of B10G11R11.
float SmallFloat(uint32_t bits, uint32_t mantissa_bits) {
	const uint32_t exp = bits >> mantissa_bits;
	const uint32_t man = bits & ((1u << mantissa_bits) - 1u);
	if (exp == 0) {
		return std::ldexp(static_cast<float>(man), -14 - static_cast<int>(mantissa_bits));
	}
	if (exp == 31) {
		return 0.0f;
	}
	return std::ldexp(1.0f + static_cast<float>(man) / static_cast<float>(1u << mantissa_bits),
	                  static_cast<int>(exp) - 15);
}

uint8_t Unorm(float v) {
	return static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// HDR values: Reinhard, so bright pixels stay visible.
uint8_t Hdr(float v) {
	v = std::max(v, 0.0f);
	return Unorm(v / (1.0f + v));
}

struct FormatInfo {
	uint32_t bytes = 0;
};

std::optional<FormatInfo> Describe(vk::Format format) {
	switch (format) {
		case vk::Format::eR8Unorm: return FormatInfo {1};
		case vk::Format::eR8G8Unorm: case vk::Format::eR16Sfloat: case vk::Format::eR16Unorm:
			return FormatInfo {2};
		case vk::Format::eR8G8B8A8Unorm: case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm: case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32: case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eB10G11R11UfloatPack32: case vk::Format::eR32Sfloat:
		case vk::Format::eR16G16Sfloat: case vk::Format::eR16G16Unorm: case vk::Format::eR32Uint:
			return FormatInfo {4};
		case vk::Format::eR16G16B16A16Sfloat: case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR32G32Sfloat:
			return FormatInfo {8};
		case vk::Format::eR32G32B32A32Sfloat: return FormatInfo {16};
		default: return std::nullopt;
	}
}

// One texel to 8-bit BGR.
void Convert(vk::Format format, const uint8_t* p, uint8_t* bgr) {
	uint8_t r = 0;
	uint8_t g = 0;
	uint8_t b = 0;
	uint32_t w = 0;
	std::memcpy(&w, p, std::min<size_t>(4, Describe(format)->bytes));
	const auto h = [&](int i) {
		uint16_t v = 0;
		std::memcpy(&v, p + i * 2, 2);
		return v;
	};
	const auto f = [&](int i) {
		float v = 0;
		std::memcpy(&v, p + i * 4, 4);
		return v;
	};
	switch (format) {
		case vk::Format::eR8Unorm: r = g = b = p[0]; break;
		case vk::Format::eR8G8Unorm: r = p[0]; g = p[1]; break;
		case vk::Format::eR16Sfloat: r = g = b = Hdr(HalfToFloat(h(0))); break;
		case vk::Format::eR16Unorm: r = g = b = static_cast<uint8_t>(h(0) >> 8u); break;
		case vk::Format::eR8G8B8A8Unorm: case vk::Format::eR8G8B8A8Srgb:
			r = p[0]; g = p[1]; b = p[2]; break;
		case vk::Format::eB8G8R8A8Unorm: case vk::Format::eB8G8R8A8Srgb:
			b = p[0]; g = p[1]; r = p[2]; break;
		case vk::Format::eA2B10G10R10UnormPack32:
			r = static_cast<uint8_t>((w & 0x3ffu) >> 2u);
			g = static_cast<uint8_t>(((w >> 10u) & 0x3ffu) >> 2u);
			b = static_cast<uint8_t>(((w >> 20u) & 0x3ffu) >> 2u);
			break;
		case vk::Format::eA2R10G10B10UnormPack32:
			b = static_cast<uint8_t>((w & 0x3ffu) >> 2u);
			g = static_cast<uint8_t>(((w >> 10u) & 0x3ffu) >> 2u);
			r = static_cast<uint8_t>(((w >> 20u) & 0x3ffu) >> 2u);
			break;
		case vk::Format::eB10G11R11UfloatPack32:
			r = Hdr(SmallFloat(w & 0x7ffu, 6));
			g = Hdr(SmallFloat((w >> 11u) & 0x7ffu, 6));
			b = Hdr(SmallFloat((w >> 22u) & 0x3ffu, 5));
			break;
		case vk::Format::eR32Sfloat: r = g = b = Hdr(f(0)); break;
		case vk::Format::eR32Uint: r = g = b = static_cast<uint8_t>(w); break;
		case vk::Format::eR16G16Sfloat: r = Hdr(HalfToFloat(h(0))); g = Hdr(HalfToFloat(h(1))); break;
		case vk::Format::eR16G16Unorm: r = static_cast<uint8_t>(h(0) >> 8u); g = static_cast<uint8_t>(h(1) >> 8u); break;
		case vk::Format::eR16G16B16A16Sfloat:
			r = Hdr(HalfToFloat(h(0))); g = Hdr(HalfToFloat(h(1))); b = Hdr(HalfToFloat(h(2))); break;
		case vk::Format::eR16G16B16A16Unorm:
			r = static_cast<uint8_t>(h(0) >> 8u); g = static_cast<uint8_t>(h(1) >> 8u);
			b = static_cast<uint8_t>(h(2) >> 8u); break;
		case vk::Format::eR32G32Sfloat: r = Hdr(f(0)); g = Hdr(f(1)); break;
		case vk::Format::eR32G32B32A32Sfloat: r = Hdr(f(0)); g = Hdr(f(1)); b = Hdr(f(2)); break;
		default: break;
	}
	bgr[0] = b;
	bgr[1] = g;
	bgr[2] = r;
}

void WriteBmp(const std::filesystem::path& path, uint32_t width, uint32_t height,
              const std::vector<uint8_t>& bgr_rows) {
	const uint32_t row  = (width * 3u + 3u) & ~3u;
	const uint32_t size = 54u + row * height;
	uint8_t        header[54] {};
	const auto     put = [&](uint32_t offset, uint32_t value, uint32_t bytes) {
		for (uint32_t i = 0; i < bytes; ++i) header[offset + i] = static_cast<uint8_t>(value >> (8u * i));
	};
	header[0] = 'B';
	header[1] = 'M';
	put(2, size, 4);
	put(10, 54, 4);
	put(14, 40, 4);
	put(18, width, 4);
	put(22, static_cast<uint32_t>(-static_cast<int32_t>(height)), 4); // top-down rows
	put(26, 1, 2);
	put(28, 24, 2);
	put(34, row * height, 4);
	std::FILE* file = std::fopen(path.string().c_str(), "wb");
	if (file == nullptr) {
		return;
	}
	std::fwrite(header, 1, sizeof(header), file);
	std::fwrite(bgr_rows.data(), 1, bgr_rows.size(), file);
	std::fclose(file);
}

bool EnsureStaging(GraphicContext& graphics, vk::DeviceSize size) {
	if (size <= g_size) {
		return true;
	}
	if (g_staging != nullptr) {
		graphics.device.destroyBuffer(g_staging, nullptr);
		graphics.device.freeMemory(g_memory, nullptr);
		g_staging = nullptr;
	}
	vk::BufferCreateInfo info {};
	info.size  = size;
	info.usage = vk::BufferUsageFlagBits::eTransferDst;
	if (graphics.device.createBuffer(&info, nullptr, &g_staging) != vk::Result::eSuccess) {
		return false;
	}
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(g_staging, &requirements);
	const auto& props  = graphics.physical_device_memory_properties;
	const auto  wanted = vk::MemoryPropertyFlagBits::eHostVisible |
	                     vk::MemoryPropertyFlagBits::eHostCoherent |
	                     vk::MemoryPropertyFlagBits::eHostCached;
	uint32_t type = UINT32_MAX;
	for (int pass = 0; pass < 2 && type == UINT32_MAX; ++pass) {
		const auto need = pass == 0 ? wanted : (wanted & ~vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eHostCached));
		for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
			if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
			    (props.memoryTypes[i].propertyFlags & need) == need) {
				type = i;
				break;
			}
		}
	}
	vk::MemoryAllocateInfo alloc {};
	alloc.allocationSize  = requirements.size;
	alloc.memoryTypeIndex = type;
	if (type == UINT32_MAX ||
	    graphics.device.allocateMemory(&alloc, nullptr, &g_memory) != vk::Result::eSuccess ||
	    graphics.device.bindBufferMemory(g_staging, g_memory, 0) != vk::Result::eSuccess ||
	    graphics.device.mapMemory(g_memory, 0, VK_WHOLE_SIZE, {}, &g_mapped) != vk::Result::eSuccess) {
		return false;
	}
	g_size = size;
	return true;
}

} // namespace

void Init() {
	const char* enabled = std::getenv("KYTY_TARGET_DUMP");
	g_enabled           = enabled != nullptr && *enabled != '\0' && *enabled != '0';
	if (!g_enabled) {
		return;
	}
	if (const char* at = std::getenv("KYTY_TARGET_DUMP_AT_FLIP")) {
		g_at_flip = static_cast<uint32_t>(std::strtoul(at, nullptr, 10));
	}
	if (const char* max = std::getenv("KYTY_TARGET_DUMP_MAX")) {
		g_max = static_cast<uint32_t>(std::strtoul(max, nullptr, 10));
	}
	if (const char* draws = std::getenv("KYTY_TARGET_DUMP_DRAWS")) {
		char* end   = nullptr;
		g_each_from = static_cast<uint32_t>(std::strtoul(draws, &end, 10));
		g_each_to   = end != nullptr && *end == '-' ? static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10))
		                                            : g_each_from;
	}
	GpuBreadcrumbs::TrackPipelines();
	Log::WriteToConsoleAndLog(fmt::format("Target dump: on (F9{})\n",
	                                      g_at_flip != 0 ? fmt::format(", at flip {}", g_at_flip) : ""));
}

bool Enabled() noexcept {
	return g_enabled;
}

void RequestCapture() noexcept {
	auto expected = State::Idle;
	if (g_enabled && g_state.compare_exchange_strong(expected, State::Requested)) {
		Log::WriteToConsoleAndLog("Target dump: requested, starts at the next flip\n");
	}
}

void StartCapture() noexcept {
	auto expected = State::Idle;
	if (g_enabled && g_state.compare_exchange_strong(expected, State::Capturing)) {
		Log::WriteToConsoleAndLog("Target dump: started at the next draw\n");
	}
}

bool NoPipelineSkip() noexcept {
	return g_enabled && g_state.load(std::memory_order_acquire) != State::Idle;
}

void NoteWaitedPipeline() noexcept {
	g_waited.fetch_add(1, std::memory_order_relaxed);
}

void OnFlip() noexcept {
	if (!g_enabled) {
		return;
	}
	if (const auto waited = g_waited.exchange(0); waited != 0) {
		Log::WriteToConsoleAndLog(fmt::format("Target dump: {} draws waited for their pipelines instead of being skipped\n", waited));
	}
	const auto flips = g_flips.fetch_add(1) + 1;
	if (g_at_flip != 0 && flips == g_at_flip) {
		RequestCapture();
	}
	// The capture ends after g_max draws, not at a flip: the dump slows the GPU work down while
	// the presentation keeps flipping.
	auto expected = State::Requested;
	g_state.compare_exchange_strong(expected, State::Capturing);
}

namespace {

struct Pending {
	uint32_t             draw = 0;
	uint32_t             slot = 0;
	uint32_t             flip = 0;
	std::string          format;
	uint32_t             guest_width  = 0;
	uint32_t             guest_height = 0;
	uint64_t             address      = 0;
	std::string          shaders;
	uint32_t             width  = 0;
	uint32_t             height = 0;
	std::vector<uint8_t> pixels;
};

using TargetKey = std::tuple<uint64_t, uint32_t, uint32_t>; // guest address, mip, layer
std::map<TargetKey, Pending> g_pending; // the last draw of each target of the current pass

void Write(const Pending& p, char prefix) {
	const auto name = fmt::format("{}{:05}_t{}.bmp", prefix, p.draw, p.slot);
	WriteBmp(g_folder / name, p.width, p.height, p.pixels);
	std::fprintf(g_index, "%u,%u,%u,%s,%u,%u,0x%llx,%s,%s\n", p.draw, p.slot, p.flip, p.format.c_str(),
	             p.guest_width, p.guest_height, static_cast<unsigned long long>(p.address), name.c_str(),
	             p.shaders.c_str());
}

void FlushExcept(const std::vector<TargetKey>& keep) {
	for (auto it = g_pending.begin(); it != g_pending.end();) {
		if (std::find(keep.begin(), keep.end(), it->first) == keep.end()) {
			Write(it->second, 'p');
			it = g_pending.erase(it);
		} else {
			++it;
		}
	}
	std::fflush(g_index);
}

} // namespace

void AfterDraw(RenderContext& context, CommandBuffer& buffer, const DrawRenderState& state,
               vk::Pipeline pipeline) {
	if (!g_enabled || g_state.load(std::memory_order_acquire) != State::Capturing) {
		return;
	}
	if (g_index == nullptr) {
		++g_frame;
		g_draw   = 0;
		g_folder = std::filesystem::path("_TargetDump") / fmt::format("frame_{}_{}", g_frame, g_flips.load());
		std::filesystem::create_directories(g_folder);
		g_index = std::fopen((g_folder / "index.csv").string().c_str(), "w");
		if (g_index == nullptr) {
			g_state.store(State::Idle);
			return;
		}
		std::fprintf(g_index, "draw,slot,flip,format,width,height,guest_address,file,shaders\n");
		Log::WriteToConsoleAndLog(fmt::format("Target dump: frame {} started ({} draws)\n", g_frame, g_max));
	}
	const uint32_t draw = ++g_draw;
	auto& graphics = context.GetGraphics();
	auto& cache    = context.GetTextureCache();
	const auto shaders = GpuBreadcrumbs::Describe(pipeline);
	std::vector<TargetKey> targets;
	std::vector<Pending>   current;
	for (uint32_t i = 0; i < state.color_count; ++i) {
		const auto& color = state.colors[i];
		if (!color.image_id) {
			continue;
		}
		auto&      image  = cache.GetImage(color.image_id);
		const auto format = image.backing.format;
		const auto info   = Describe(format);
		const auto extent = color.Extent();
		const auto mip    = color.desc.view_info.base_level;
		const auto layer  = color.desc.view_info.base_layer;
		if (!info || image.backing.samples > 1 ||
		    !EnsureStaging(graphics, static_cast<vk::DeviceSize>(extent.width) * extent.height * info->bytes)) {
			std::fprintf(g_index, "%u,%u,%u,%d,%u,%u,0x%llx,,%s\n", draw, color.target_slot, g_flips.load(),
			             static_cast<int>(format), extent.width, extent.height,
			             static_cast<unsigned long long>(image.info.data.address), shaders.c_str());
			continue;
		}
		buffer.EndRendering();
		const auto command = buffer.Handle();
		image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
		vk::BufferImageCopy region {};
		region.imageSubresource = {vk::ImageAspectFlagBits::eColor, mip, layer, 1};
		region.imageExtent      = vk::Extent3D {extent.width, extent.height, 1};
		command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, g_staging, 1,
		                          &region);
		if (image.binding.attachment_layout != vk::ImageLayout::eUndefined) {
			image.Transit(image.binding.attachment_layout, image.binding.attachment_access, {}, command);
		}
		auto& scheduler = context.GetCommandScheduler();
		scheduler.Wait(scheduler.CurrentTick());

		const uint32_t step = std::max(1u, (extent.width + 1279u) / 1280u);
		Pending        p;
		p.draw         = draw;
		p.slot         = color.target_slot;
		p.flip         = g_flips.load();
		p.format       = vk::to_string(format);
		p.guest_width  = extent.width;
		p.guest_height = extent.height;
		p.address      = image.info.data.address;
		p.shaders      = shaders;
		p.width        = (extent.width + step - 1) / step;
		p.height       = (extent.height + step - 1) / step;
		const uint32_t row = (p.width * 3u + 3u) & ~3u;
		p.pixels.resize(static_cast<size_t>(row) * p.height);
		const auto* source = static_cast<const uint8_t*>(g_mapped);
		for (uint32_t y = 0; y < p.height; ++y) {
			for (uint32_t x = 0; x < p.width; ++x) {
				const size_t texel =
				    (static_cast<size_t>(y * step) * extent.width + x * step) * info->bytes;
				Convert(format, source + texel, &p.pixels[static_cast<size_t>(y) * row + x * 3u]);
			}
		}
		if (draw >= g_each_from && draw <= g_each_to) {
			Write(p, 'd');
		}
		targets.emplace_back(p.address, mip, layer);
		current.push_back(std::move(p));
	}
	// A draw to other targets ends the pass of the targets it does not use: their last draw's
	// image is final for that pass.
	// Depth-only draws (shadow maps, prepasses) leave the color passes open.
	if (!targets.empty()) {
		FlushExcept(targets);
	}
	for (size_t i = 0; i < current.size(); ++i) {
		g_pending[targets[i]] = std::move(current[i]);
	}
	if (draw >= g_max) {
		FlushExcept({});
		std::fclose(g_index);
		g_index = nullptr;
		g_state.store(State::Idle);
		Log::WriteToConsoleAndLog(fmt::format("Target dump: frame {} done, {} draws in {}\n", g_frame,
		                                      draw, g_folder.string()));
	}
}

} // namespace Libs::Graphics::TargetDump