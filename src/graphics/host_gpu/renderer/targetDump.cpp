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
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace Libs::Graphics::TargetDump {

namespace {

enum class State : uint32_t { Idle, Requested, Capturing };

bool                  g_enabled  = false;
uint32_t              g_at_flip  = 0;
uint32_t              g_max      = 50000; // a safety stop; the frame ends at the next stream flip
uint32_t              g_each_from = 1;   // draws dumped one by one (KYTY_TARGET_DUMP_DRAWS=a-b)
uint32_t              g_each_to   = 0;
std::atomic<State>    g_state {State::Idle};
std::atomic<uint32_t> g_flips {0};
std::atomic<uint32_t> g_waited {0};
uint32_t              g_frame    = 0;
uint32_t              g_draw     = 0;
std::FILE*            g_index    = nullptr;
std::filesystem::path g_folder;
std::filesystem::path g_trigger; // KYTY_TARGET_DUMP_TRIGGER

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
		case vk::Format::eR8Unorm: case vk::Format::eR8Uint: return FormatInfo {1};
		case vk::Format::eR8G8Unorm: case vk::Format::eR16Sfloat: case vk::Format::eR16Unorm: case vk::Format::eR16Uint:
			return FormatInfo {2};
		case vk::Format::eR8G8B8A8Unorm: case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm: case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32: case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eB10G11R11UfloatPack32: case vk::Format::eR32Sfloat:
		case vk::Format::eR16G16Sfloat: case vk::Format::eR16G16Unorm: case vk::Format::eR32Uint:
			return FormatInfo {4};
		case vk::Format::eR16G16B16A16Sfloat: case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR32G32Sfloat: case vk::Format::eR32G32Uint:
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
		case vk::Format::eR8Uint: r = g = b = p[0]; break;
		case vk::Format::eR16Uint: r = g = b = static_cast<uint8_t>(h(0)); break;
		// Light or tile bit masks: 16 levels per set bit.
		case vk::Format::eR32G32Uint: {
			uint32_t x = 0;
			uint32_t y = 0;
			std::memcpy(&x, p, 4);
			std::memcpy(&y, p + 4, 4);
			r = static_cast<uint8_t>(std::min(255, std::popcount(x) * 16));
			g = static_cast<uint8_t>(std::min(255, std::popcount(y) * 16));
			break;
		}
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

// The resolve target of a multisampled color image (vkCmdCopyImageToBuffer takes one sample).
struct ResolveImage {
	vk::Image        image  = nullptr;
	vk::DeviceMemory memory = nullptr;
	vk::Format       format = vk::Format::eUndefined;
	vk::Extent2D     extent {};
};
ResolveImage g_resolve;

// Called with the GPU idle (each read back waits for it), so the old image can go at once.
bool EnsureResolveImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent) {
	if (g_resolve.image != nullptr && g_resolve.format == format && g_resolve.extent == extent) {
		return true;
	}
	if (g_resolve.image != nullptr) {
		graphics.device.destroyImage(g_resolve.image, nullptr);
		graphics.device.freeMemory(g_resolve.memory, nullptr);
		g_resolve = {};
	}
	vk::ImageCreateInfo info {};
	info.imageType     = vk::ImageType::e2D;
	info.format        = format;
	info.extent        = vk::Extent3D {extent.width, extent.height, 1};
	info.mipLevels     = 1;
	info.arrayLayers   = 1;
	info.samples       = vk::SampleCountFlagBits::e1;
	info.tiling        = vk::ImageTiling::eOptimal;
	info.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
	info.initialLayout = vk::ImageLayout::eUndefined;
	if (graphics.device.createImage(&info, nullptr, &g_resolve.image) != vk::Result::eSuccess) {
		g_resolve.image = nullptr;
		return false;
	}
	vk::MemoryRequirements requirements {};
	graphics.device.getImageMemoryRequirements(g_resolve.image, &requirements);
	const auto& props = graphics.physical_device_memory_properties;
	uint32_t    type  = UINT32_MAX;
	for (uint32_t i = 0; i < props.memoryTypeCount && type == UINT32_MAX; ++i) {
		if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
		    (props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
			type = i;
		}
	}
	vk::MemoryAllocateInfo alloc {};
	alloc.allocationSize  = requirements.size;
	alloc.memoryTypeIndex = type;
	if (type == UINT32_MAX ||
	    graphics.device.allocateMemory(&alloc, nullptr, &g_resolve.memory) != vk::Result::eSuccess ||
	    graphics.device.bindImageMemory(g_resolve.image, g_resolve.memory, 0) != vk::Result::eSuccess) {
		graphics.device.destroyImage(g_resolve.image, nullptr);
		if (g_resolve.memory != nullptr) {
			graphics.device.freeMemory(g_resolve.memory, nullptr);
		}
		g_resolve = {};
		return false;
	}
	g_resolve.format = format;
	g_resolve.extent = extent;
	return true;
}

void ResolveBarrier(CommandRecorder command, vk::ImageLayout from, vk::ImageLayout to,
                    vk::AccessFlags src, vk::AccessFlags dst) {
	vk::ImageMemoryBarrier barrier {};
	barrier.srcAccessMask       = src;
	barrier.dstAccessMask       = dst;
	barrier.oldLayout           = from;
	barrier.newLayout           = to;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = g_resolve.image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
	                        {}, 0, nullptr, 0, nullptr, 1, &barrier);
}

// Depth texel bytes of the depth aspect copy, or 0.
uint32_t DepthBytes(vk::Format format) {
	switch (format) {
		case vk::Format::eD16Unorm: case vk::Format::eD16UnormS8Uint: return 2;
		case vk::Format::eD32Sfloat: case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eX8D24UnormPack32: case vk::Format::eD24UnormS8Uint: return 4;
		default: return 0;
	}
}

float DepthValue(vk::Format format, const uint8_t* p) {
	uint32_t w = 0;
	std::memcpy(&w, p, DepthBytes(format));
	switch (format) {
		case vk::Format::eD16Unorm: case vk::Format::eD16UnormS8Uint:
			return static_cast<float>(w & 0xffffu) / 65535.0f;
		case vk::Format::eX8D24UnormPack32: case vk::Format::eD24UnormS8Uint:
			return static_cast<float>(w & 0xffffffu) / 16777215.0f;
		default: {
			float v = 0.0f;
			std::memcpy(&v, p, 4);
			return v;
		}
	}
}

enum class Kind : char { Color = 'c', Depth = 'z', Storage = 's' };

struct Pending {
	uint32_t             op   = 0;
	Kind                 kind = Kind::Color;
	uint32_t             slot = 0;
	uint32_t             flip = 0;
	std::string          format;
	uint32_t             guest_width  = 0;
	uint32_t             guest_height = 0;
	uint32_t             samples      = 1;
	uint32_t             mip          = 0;
	uint32_t             layer        = 0;
	uint64_t             address      = 0;
	std::string          shaders;
	std::string          note; // depth range, or why there is no image
	uint32_t             width  = 0;
	uint32_t             height = 0;
	std::vector<uint8_t> pixels;
};

// guest address, mip, layer, kind
using TargetKey = std::tuple<uint64_t, uint32_t, uint32_t, char>;
std::map<TargetKey, Pending> g_pending; // the last operation on each target of the current pass

void WriteRow(const Pending& p, const std::string& file) {
	std::fprintf(g_index, "%u,%c,%u,%u,%s,%u,%u,%u,%u,%u,0x%llx,%s,%s,%s\n", p.op,
	             static_cast<char>(p.kind), p.slot, p.flip, p.format.c_str(), p.guest_width,
	             p.guest_height, p.samples, p.mip, p.layer, static_cast<unsigned long long>(p.address),
	             file.c_str(), p.shaders.c_str(), p.note.c_str());
}

void Write(const Pending& p, char prefix) {
	const auto name = fmt::format("{}{:05}_{}{}.bmp", prefix, p.op, static_cast<char>(p.kind), p.slot);
	WriteBmp(g_folder / name, p.width, p.height, p.pixels);
	WriteRow(p, name);
}

// An operation on targets of a group ends the pass of the other targets of that group it does
// not use: their last operation's image is final for that pass. Draws (color, depth) and
// dispatches (storage) are separate groups, so a dispatch between draws leaves the draw passes open.
void FlushExcept(const std::vector<TargetKey>& keep, bool storage_group) {
	for (auto it = g_pending.begin(); it != g_pending.end();) {
		const bool storage = std::get<3>(it->first) == static_cast<char>(Kind::Storage);
		if (storage == storage_group && std::find(keep.begin(), keep.end(), it->first) == keep.end()) {
			Write(it->second, 'p');
			it = g_pending.erase(it);
		} else {
			++it;
		}
	}
	std::fflush(g_index);
}

void FlushAll() {
	FlushExcept({}, false);
	FlushExcept({}, true);
}

void Finish(const char* why) {
	FlushAll();
	std::fclose(g_index);
	g_index = nullptr;
	g_state.store(State::Idle);
	Log::WriteToConsoleAndLog(fmt::format("Target dump: frame {} done ({}), {} operations in {}\n", g_frame,
	                                      why, g_draw, g_folder.string()));
}

bool BeginOperation() {
	if (!g_enabled || g_state.load(std::memory_order_acquire) != State::Capturing) {
		return false;
	}
	if (g_index == nullptr) {
		++g_frame;
		g_draw   = 0;
		g_folder = std::filesystem::path("_TargetDump") / fmt::format("frame_{}_{}", g_frame, g_flips.load());
		std::filesystem::create_directories(g_folder);
		g_index = std::fopen((g_folder / "index.csv").string().c_str(), "w");
		if (g_index == nullptr) {
			g_state.store(State::Idle);
			return false;
		}
		std::fprintf(g_index,
		             "op,kind,slot,flip,format,width,height,samples,mip,layer,guest_address,file,shaders,note\n");
		Log::WriteToConsoleAndLog(fmt::format("Target dump: frame {} started at a stream flip\n", g_frame));
	}
	return true;
}

// Reads back one subresource of a cached image after the operation recorded so far.
// Returns false (with p.note) when it has no image.
bool ReadBack(RenderContext& context, CommandBuffer& buffer, Image& image, Kind kind,
              vk::Extent2D extent, Pending& p) {
	auto&      graphics = context.GetGraphics();
	const auto format   = image.backing.format;
	const bool depth    = kind == Kind::Depth;
	const auto info     = depth ? std::optional<FormatInfo>(FormatInfo {DepthBytes(format)}) : Describe(format);
	p.format  = vk::to_string(format);
	p.samples = image.backing.samples;
	if (image.backing.image_type == vk::ImageType::e3D) {
		p.note  = "3d: slice 0";
		p.layer = 0;
	}
	if (!info || info->bytes == 0) {
		p.note = "format not converted";
		return false;
	}
	if (image.backing.samples > 1 && depth) {
		p.note = "multisampled depth";
		return false;
	}
	if (p.mip >= image.backing.mip_levels ||
	    (image.backing.image_type != vk::ImageType::e3D && p.layer >= image.backing.layers)) {
		p.note = fmt::format("view outside image (mips {} layers {})", image.backing.mip_levels,
		                     image.backing.layers);
		return false;
	}
	extent.width  = std::min(extent.width, std::max(image.backing.extent.width >> p.mip, 1u));
	extent.height = std::min(extent.height, std::max(image.backing.extent.height >> p.mip, 1u));
	if (!EnsureStaging(graphics, static_cast<vk::DeviceSize>(extent.width) * extent.height * info->bytes)) {
		p.note = "staging allocation failed";
		return false;
	}
	const bool resolve = image.backing.samples > 1;
	if (resolve && !EnsureResolveImage(graphics, format, extent)) {
		p.note = "resolve image allocation failed";
		return false;
	}
	buffer.EndRendering();
	const auto command = buffer.Handle();
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	vk::BufferImageCopy region {};
	region.imageExtent = vk::Extent3D {extent.width, extent.height, 1};
	if (resolve) {
		ResolveBarrier(command, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {},
		               vk::AccessFlagBits::eTransferWrite);
		vk::ImageResolve resolve_region {};
		resolve_region.srcSubresource = {vk::ImageAspectFlagBits::eColor, p.mip, p.layer, 1};
		resolve_region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		resolve_region.extent         = region.imageExtent;
		command.resolveImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, g_resolve.image,
		                     vk::ImageLayout::eTransferDstOptimal, 1, &resolve_region);
		ResolveBarrier(command, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
		               vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead);
		region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		command.copyImageToBuffer(g_resolve.image, vk::ImageLayout::eTransferSrcOptimal, g_staging, 1, &region);
	} else {
		const bool volume = image.backing.image_type == vk::ImageType::e3D;
		region.imageSubresource = {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
		                           p.mip, volume ? 0u : p.layer, 1};
		command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, g_staging, 1,
		                          &region);
	}
	if (image.binding.attachment_layout != vk::ImageLayout::eUndefined) {
		image.Transit(image.binding.attachment_layout, image.binding.attachment_access, {}, command);
	}
	auto& scheduler = context.GetCommandScheduler();
	scheduler.Wait(scheduler.CurrentTick());

	const uint32_t step = std::max(1u, (extent.width + 1279u) / 1280u);
	p.width  = (extent.width + step - 1) / step;
	p.height = (extent.height + step - 1) / step;
	const uint32_t row = (p.width * 3u + 3u) & ~3u;
	p.pixels.resize(static_cast<size_t>(row) * p.height);
	const auto* source = static_cast<const uint8_t*>(g_mapped);
	const auto  texel  = [&](uint32_t x, uint32_t y) {
        return source + (static_cast<size_t>(y * step) * extent.width + x * step) * info->bytes;
	};
	if (depth) {
		// Stretched to the range the image holds: depth sits in a narrow band (reversed Z).
		float lo = std::numeric_limits<float>::max();
		float hi = std::numeric_limits<float>::lowest();
		for (uint32_t y = 0; y < p.height; ++y) {
			for (uint32_t x = 0; x < p.width; ++x) {
				const float v = DepthValue(format, texel(x, y));
				lo            = std::min(lo, v);
				hi            = std::max(hi, v);
			}
		}
		const float scale = hi > lo ? 1.0f / (hi - lo) : 0.0f;
		for (uint32_t y = 0; y < p.height; ++y) {
			for (uint32_t x = 0; x < p.width; ++x) {
				auto* bgr = &p.pixels[static_cast<size_t>(y) * row + x * 3u];
				bgr[0] = bgr[1] = bgr[2] = Unorm((DepthValue(format, texel(x, y)) - lo) * scale);
			}
		}
		p.note = fmt::format("depth {:.6f}..{:.6f}", lo, hi);
	} else {
		for (uint32_t y = 0; y < p.height; ++y) {
			for (uint32_t x = 0; x < p.width; ++x) {
				Convert(format, texel(x, y), &p.pixels[static_cast<size_t>(y) * row + x * 3u]);
			}
		}
		if (resolve) {
			p.note = "resolved";
		}
	}
	return true;
}

// Reads back the targets of one operation and moves them to the pass bookkeeping.
struct Target {
	ImageId      id;
	Kind         kind = Kind::Color;
	uint32_t     slot = 0;
	uint32_t     mip  = 0;
	uint32_t     layer = 0;
	vk::Extent2D extent {};
};

void DumpTargets(RenderContext& context, CommandBuffer& buffer, std::span<const Target> list,
                 vk::Pipeline pipeline, bool storage_group) {
	const uint32_t op      = ++g_draw;
	auto&          cache   = context.GetTextureCache();
	const auto     shaders = GpuBreadcrumbs::Describe(pipeline);
	std::vector<TargetKey> keys;
	std::vector<Pending>   current;
	for (const auto& target : list) {
		auto&   image = cache.GetImage(target.id);
		Pending p;
		p.op           = op;
		p.kind         = target.kind;
		p.slot         = target.slot;
		p.flip         = g_flips.load();
		p.guest_width  = target.extent.width;
		p.guest_height = target.extent.height;
		p.mip          = target.mip;
		p.layer        = target.layer;
		p.address      = image.info.data.address;
		p.shaders      = shaders;
		if (!ReadBack(context, buffer, image, target.kind, target.extent, p)) {
			WriteRow(p, "");
			continue;
		}
		if (op >= g_each_from && op <= g_each_to) {
			Write(p, 'd');
		}
		keys.emplace_back(p.address, p.mip, p.layer, static_cast<char>(p.kind));
		current.push_back(std::move(p));
	}
	// Depth-only draws (shadow maps, prepasses) now have a depth key, so they end color passes
	// too, as any draw to other targets does.
	if (!keys.empty()) {
		FlushExcept(keys, storage_group);
	}
	for (size_t i = 0; i < current.size(); ++i) {
		g_pending[keys[i]] = std::move(current[i]);
	}
	if (op >= g_max) {
		Finish("operation limit");
	}
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
	if (const char* trigger = std::getenv("KYTY_TARGET_DUMP_TRIGGER"); trigger != nullptr && *trigger != '\0') {
		g_trigger = trigger;
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
	Log::WriteToConsoleAndLog(fmt::format(
	    "Target dump: on (F9{}{})\n", g_at_flip != 0 ? fmt::format(", at flip {}", g_at_flip) : "",
	    g_trigger.empty() ? "" : fmt::format(", trigger file {}", g_trigger.string())));
}

bool Enabled() noexcept {
	return g_enabled;
}

void RequestCapture() noexcept {
	auto expected = State::Idle;
	if (g_enabled && g_state.compare_exchange_strong(expected, State::Requested)) {
		Log::WriteToConsoleAndLog("Target dump: requested, starts at the next flip in the command stream\n");
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
	// Polled twice a second at 60 fps; the file is removed so one file is one capture.
	if (!g_trigger.empty() && flips % 30 == 0) {
		std::error_code error;
		if (std::filesystem::exists(g_trigger, error)) {
			std::filesystem::remove(g_trigger, error);
			RequestCapture();
		}
	}
}

void OnStreamFlip() noexcept {
	if (!g_enabled) {
		return;
	}
	auto expected = State::Requested;
	if (g_state.compare_exchange_strong(expected, State::Capturing)) {
		return;
	}
	if (expected == State::Capturing) {
		if (g_index != nullptr) {
			Finish("next stream flip");
		} else {
			g_state.store(State::Idle);
		}
	}
}

void AfterDraw(RenderContext& context, CommandBuffer& buffer, const DrawRenderState& state,
               vk::Pipeline pipeline) {
	if (!BeginOperation()) {
		return;
	}
	std::vector<Target> list;
	for (uint32_t i = 0; i < state.color_count; ++i) {
		const auto& color = state.colors[i];
		if (color.image_id) {
			list.push_back({color.image_id, Kind::Color, color.target_slot, color.desc.view_info.base_level,
			                color.desc.view_info.base_layer, color.Extent()});
		}
	}
	const auto* depth = state.depth;
	if (depth != nullptr && depth->image_id && depth->depth_write_enable) {
		const auto mip = depth->desc.view_info.base_level;
		list.push_back({depth->image_id, Kind::Depth, 0, mip, depth->desc.view_info.base_layer,
		                {std::max(depth->desc.info.extent.width >> mip, 1u),
		                 std::max(depth->desc.info.extent.height >> mip, 1u)}});
	}
	DumpTargets(context, buffer, list, pipeline, false);
}

void AfterDispatch(RenderContext& context, CommandBuffer& buffer, std::span<const StorageTarget> targets,
                   vk::Pipeline pipeline) {
	if (!BeginOperation()) {
		return;
	}
	auto&               cache = context.GetTextureCache();
	std::vector<Target> list;
	for (uint32_t i = 0; i < targets.size(); ++i) {
		const ImageId id {targets[i].id};
		const auto&   image = cache.GetImage(id);
		const auto    mip   = targets[i].mip;
		list.push_back({id, Kind::Storage, i, mip, targets[i].layer,
		                {std::max(image.backing.extent.width >> mip, 1u),
		                 std::max(image.backing.extent.height >> mip, 1u)}});
	}
	DumpTargets(context, buffer, list, pipeline, true);
}

} // namespace Libs::Graphics::TargetDump
