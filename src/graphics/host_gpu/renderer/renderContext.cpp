#include "graphics/host_gpu/renderer/renderContext.h"
#include "hybrid/hybrid.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/hackNotice.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shaderBindings.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_graphics(graphics), m_render_executor(*this),
      m_command_scheduler(*this, graphics,
                          (Config::GpuPipelineStages() & Config::GPU_PIPELINE_RECORDING) != 0),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics, &m_image_heap, shader_store), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

namespace {

// KYTY_FAULT_STATS=1: a [kyty-faults] line every 10 s with the faults the caches resolve: writes and
// reads, the cache that watched a written page, how many frames in a row each faulting page
// faulted, and the handler's own time. Dry: it changes nothing the caches do.
struct FaultStats {
	struct Page {
		int      frame  = -1;
		uint32_t streak = 0;
	};
	std::mutex                            mutex;
	std::unordered_map<uint64_t, Page>    pages;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	uint64_t writes = 0, reads = 0, buffer = 0, texture = 0, both = 0, neither = 0;
	uint64_t streak[4] {}; // write faults on pages faulting 1, 2-3, 4-9, >= 10 frames in a row
	uint64_t handler_ns = 0;
	uint64_t distinct   = 0;
};

bool FaultStatsOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FAULT_STATS");
		return value != nullptr && *value != '\0' && *value != '0';
	}();
	return on;
}

FaultStats g_fault_stats;

struct StageTotals {
	std::mutex            mutex;
	Hybrid::FaultStageTime sum;
};
StageTotals g_stage_totals;

void RecordStages(const Hybrid::FaultStageTime& before, const Hybrid::FaultStageTime& after) {
	std::scoped_lock lock(g_stage_totals.mutex);
	auto& s = g_stage_totals.sum;
	s.protect_ns += after.protect_ns - before.protect_ns;
	s.protect_calls += after.protect_calls - before.protect_calls;
	s.readback_ns += after.readback_ns - before.readback_ns;
	s.readbacks += after.readbacks - before.readbacks;
	s.texture_lock_ns += after.texture_lock_ns - before.texture_lock_ns;
	s.buffer_ns += after.buffer_ns - before.buffer_ns;
	s.texture_ns += after.texture_ns - before.texture_ns;
	s.heap_ns += after.heap_ns - before.heap_ns;
}

void RecordFault(bool write, uint64_t vaddr, int frame, bool in_buffer, bool in_texture, uint64_t ns) {
	auto&      f = g_fault_stats;
	std::scoped_lock lock(f.mutex);
	f.handler_ns += ns;
	if (!write) {
		++f.reads;
	} else {
		++f.writes;
		(in_buffer && in_texture ? f.both : in_buffer ? f.buffer : in_texture ? f.texture : f.neither)++;
		auto& page = f.pages[vaddr / TRACKER_PAGE_SIZE];
		if (page.frame != frame) {
			if (page.frame < 0) {
				++f.distinct;
			}
			page.streak = page.frame == frame - 1 ? page.streak + 1 : 1;
			page.frame  = frame;
		}
		f.streak[page.streak >= 10 ? 3 : page.streak >= 4 ? 2 : page.streak >= 2 ? 1 : 0]++;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - f.start < std::chrono::seconds(10)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - f.start).count();
	const auto   faults  = f.writes + f.reads;
	std::fprintf(stderr,
	             "[kyty-faults] %.1f s: write %llu read %llu; write owner buffer %llu texture %llu both %llu "
	             "other %llu; write by frames in a row 1:%llu 2-3:%llu 4-9:%llu >=10:%llu (>=4: %.1f%%); "
	             "new pages %llu; handler %.2f us/fault\n",
	             seconds, static_cast<unsigned long long>(f.writes), static_cast<unsigned long long>(f.reads),
	             static_cast<unsigned long long>(f.buffer), static_cast<unsigned long long>(f.texture),
	             static_cast<unsigned long long>(f.both), static_cast<unsigned long long>(f.neither),
	             static_cast<unsigned long long>(f.streak[0]), static_cast<unsigned long long>(f.streak[1]),
	             static_cast<unsigned long long>(f.streak[2]), static_cast<unsigned long long>(f.streak[3]),
	             f.writes != 0 ? 100.0 * static_cast<double>(f.streak[2] + f.streak[3]) / static_cast<double>(f.writes) : 0.0,
	             static_cast<unsigned long long>(f.distinct),
	             faults != 0 ? static_cast<double>(f.handler_ns) / 1e3 / static_cast<double>(faults) : 0.0);
	{
		std::scoped_lock stage_lock(g_stage_totals.mutex);
		auto&            s  = g_stage_totals.sum;
		const auto       us = [&](uint64_t ns) { return f.writes != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(f.writes) : 0.0; };
		std::fprintf(stderr,
		             "[kyty-fault-stages] per write fault: buffer %.2f us (readback %.2f us, %llu readbacks), texture %.2f us "
		             "(lock wait %.2f us), image heap %.2f us; protect_host %.2f us in %llu calls\n",
		             us(s.buffer_ns), us(s.readback_ns), static_cast<unsigned long long>(s.readbacks), us(s.texture_ns),
		             us(s.texture_lock_ns), us(s.heap_ns), us(s.protect_ns), static_cast<unsigned long long>(s.protect_calls));
		s = {};
	}
	f.start  = now;
	f.writes = f.reads = f.buffer = f.texture = f.both = f.neither = f.handler_ns = f.distinct = 0;
	std::fill(std::begin(f.streak), std::end(f.streak), 0);
}

} // namespace

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	if (FaultStatsOn()) {
		const auto start = std::chrono::steady_clock::now();
		const bool write = access == PageFaultAccess::Write;
		const bool in_buffer  = write && m_buffer_cache.IsRegionRegistered(fault_vaddr & ~(TRACKER_PAGE_SIZE - 1), TRACKER_PAGE_SIZE);
		const bool in_texture = write && m_texture_cache.IsRegionRegistered(fault_vaddr & ~(TRACKER_PAGE_SIZE - 1), TRACKER_PAGE_SIZE);
		const auto before = Hybrid::g_fault_stage_time;
		const bool handled = HandleFaultUncounted(access, fault_vaddr);
		if (handled) {
			RecordStages(before, Hybrid::g_fault_stage_time);
		}
		if (handled) {
			RecordFault(write, fault_vaddr, m_gpu != nullptr ? m_gpu->GetFrameNum() : 0, in_buffer, in_texture,
			            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                      std::chrono::steady_clock::now() - start)
			                                      .count()));
		}
		return handled;
	}
	return HandleFaultUncounted(access, fault_vaddr);
}

bool RenderContext::HandleFaultUncounted(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		// Cache tracking may still protect a page whose range left the GPU-mapped set; that
		// fault is ours to resolve, not a guest access violation.
		if (!GuestRange {fault_vaddr, fault_size}.Valid() || !m_page_manager.IsWatched(fault_vaddr)) {
			return false;
		}
	}
	if (access == PageFaultAccess::Write && GuestGpu::IsGpuThread()) {
		// A write of the command stream: device-address readers after it must see it.
		m_bda_generation++;
	}
	if (access == PageFaultAccess::Write && FaultStatsOn()) {
		auto&      stage = Hybrid::g_fault_stage_time;
		const auto t0    = Hybrid::FaultStageClockNs();
		m_buffer_cache.InvalidateFaultedPage(fault_vaddr);
		const auto t1 = Hybrid::FaultStageClockNs();
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		const auto t2 = Hybrid::FaultStageClockNs();
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
		stage.buffer_ns += t1 - t0;
		stage.texture_ns += t2 - t1;
		stage.heap_ns += Hybrid::FaultStageClockNs() - t2;
	} else if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateFaultedPage(fault_vaddr);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size, false, true);
		m_texture_cache.FlushLazyReadback(fault_vaddr, fault_size);
	}
	return true;
}

void RenderContext::WriteFromCompletion(void* destination, const void* data,
                                        uint64_t size) noexcept {
	const auto vaddr = reinterpret_cast<uint64_t>(destination);
	// Through the backing store the bytes never fault, even on a page the execution thread
	// hands back to the GPU while the completion runs. The caches then learn of the write as from
	// a write fault, except that pages the GPU owns stay GPU-owned (a readback of them downloads
	// only the bytes the GPU wrote).
	if (GuestRange {vaddr, size}.Valid() && LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		const auto page_size = TRACKER_PAGE_SIZE;
		for (auto page = vaddr & ~(page_size - 1); page < vaddr + size; page += page_size) {
			if (m_page_manager.IsWatched(page)) {
				m_buffer_cache.InvalidateUnlessGpuModified(vaddr, size);
				m_texture_cache.InvalidateMemory(vaddr, size);
				m_image_heap.InvalidateMemory(vaddr, size);
				break;
			}
		}
		return;
	}
	if (size == sizeof(uint32_t)) {
		uint32_t value = 0;
		std::memcpy(&value, data, sizeof(value));
		*static_cast<volatile uint32_t*>(destination) = value;
	} else if (size == sizeof(uint64_t)) {
		uint64_t value = 0;
		std::memcpy(&value, data, sizeof(value));
		*static_cast<volatile uint64_t*>(destination) = value;
	} else {
		std::memcpy(destination, data, size);
	}
}

void RenderContext::PrepareGpuWrite(const void* destination, uint64_t size) noexcept {
	const auto vaddr     = reinterpret_cast<uint64_t>(destination);
	const auto page_size = TRACKER_PAGE_SIZE;
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// A watched page is protected: the write would fault into HandleFault().
	for (auto page = vaddr & ~(page_size - 1); page < vaddr + size; page += page_size) {
		if (m_page_manager.IsWatched(page)) {
			(void)HandleFault(PageFaultAccess::Write, std::max(page, vaddr));
		}
	}
}

bool RenderContext::WriteAroundGpuWrites(void* destination, const void* data, uint64_t size) {
	const auto vaddr = reinterpret_cast<uint64_t>(destination);
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	if (m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		// A write over memory a GPU-written image holds: the faulting path reads the GPU's
		// buffer writes of the page back (draining the GPU) and leaves the image CPU-dirty. The
		// write-around keeps the page GPU-owned instead, which needs no readback, and the image
		// becomes CPU-dirty all the same.
		if (!m_buffer_cache.WriteAroundGpuWrites(vaddr, data, size)) {
			return false;
		}
		m_texture_cache.InvalidateMemory(vaddr, size);
		return true;
	}
	return m_buffer_cache.WriteAroundGpuWrites(vaddr, data, size);
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

uint64_t RenderContext::MappedRangesVersion() const noexcept {
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges_version;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_mapped_ranges_version++;
	m_bda_mapped_since_sync.emplace_back(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Finishing the GPU protects GPU writes to the range and completions that may write guest
		// memory there (downloads, write scans). Queued GPU work only reads cached copies and
		// host-only completions (cache bookkeeping, flips, interrupts) never touch the range, so
		// without GPU-owned pages, images or pending guest completions writing the range the
		// unmap needs no drain.
		if (m_command_scheduler.Active()) {
			m_command_scheduler.PopPendingOperations();
		}
		const bool drain = m_command_scheduler.Active() &&
		                   (m_buffer_cache.IsRegionGpuModified(vaddr, size) ||
		                    m_texture_cache.IsRegionRegistered(vaddr, size) ||
		                    m_command_scheduler.HasPendingGuestOperations(vaddr, size));
		if (drain) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		m_occlusion_queries.OnUnmap(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_mapped_ranges_version++;
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	// The guest unmaps memory once the GPU work that used it executed: it waits on labels the
	// execution thread writes after the operations before them. The operations queued since do not
	// use the range, so the unmap runs before them instead of making the guest wait for them;
	// streaming games unmap many times per frame, holding their own locks.
	m_gpu->SendUrgentCommandSync(unmap);
}

void RenderContext::CacheDmaBases(const ShaderStageRuntime& runtime) {
	const auto&      program   = runtime.module->info;
	const auto&      user_data = runtime.resources->user_data;
	std::shared_lock lock(m_mapped_ranges_mutex);
	for (const auto reg: program.info.dma_base_registers) {
		const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
		if (reg < program.user_data_base || index + 1u >= user_data.size()) {
			continue;
		}
		const auto base = user_data[index] | (static_cast<uint64_t>(user_data[index + 1u]) << 32u);
		// The registers of an access that never runs may hold stale data. Mapped memory never
		// reaches the top of the address space, so a window that would overflow is skipped.
		if (base > UINT64_MAX - BufferCache::CACHING_PAGESIZE) {
			continue;
		}
		// DMA reaches only memory with a cached buffer; any other access records a fault and
		// reads zero, and the buffer arrives after the shader has run. That loses data the guest
		// writes for a single dispatch, such as the glyph bitmaps the title copies into its font atlas.
		m_mapped_ranges.ForEachInRange(base, BufferCache::CACHING_PAGESIZE,
		                               [this](uint64_t start, uint64_t end) {
			                               (void)m_buffer_cache.FindBuffer(start, end - start);
		                               });
	}
}

bool RenderContext::SynchronizeDmaFootprint(std::span<const ShaderStageRuntime* const> stages) {
	// Larger V# ranges are no buffers a draw reads (see FindBuffers()).
	constexpr uint64_t MaxBufferBytes = uint64_t {256} * 1024 * 1024;
	// An index offset or thread ID added to a V# record stays within a wave of records past it.
	constexpr uint64_t WaveRecords = 64;
	m_dma_spans.clear();
	std::shared_lock lock(m_mapped_ranges_mutex);
	const auto add = [&](uint64_t address, uint64_t size) {
		if (size != 0 && address <= UINT64_MAX - size) {
			m_mapped_ranges.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
				m_dma_spans.emplace_back(start, end);
			});
		}
	};
	for (const auto* stage: stages) {
		const auto& program = stage->module->info;
		if (!program.info.uses_dma) {
			continue;
		}
		if (!program.info.dma_bounded) {
			return false;
		}
		const auto& user_data = stage->resources->user_data;
		const auto  base_of   = [&](uint32_t reg, uint64_t& base) {
			const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
			if (reg < program.user_data_base || index + 1u >= user_data.size()) {
				return false;
			}
			base = user_data[index] | (static_cast<uint64_t>(user_data[index + 1u]) << 32u);
			return true;
		};
		for (const auto& window: program.info.dma_windows) {
			uint64_t base = 0;
			if (!base_of(window.base_register, base)) {
				return false;
			}
			// Scalar address reads clear the low two bits of the base.
			const auto before = static_cast<uint64_t>(-window.first) + 4u;
			const auto start  = base > before ? base - before : 0;
			add(start, std::min(base, UINT64_MAX - window.last) + window.last - start);
		}
		for (const auto& table: program.info.dma_tables) {
			uint64_t base = 0;
			if (!base_of(table.base_register, base)) {
				return false;
			}
			const auto entries = (base & ~uint64_t {3}) + static_cast<uint64_t>(table.immediate);
			Common::HackNotice::Report("DMA upload assumes V# indices stay within 64 records",
			                           "RenderContext::SynchronizeDmaFootprint",
			                           program.shader_hash, table.base_register);
			for (uint32_t offset = table.offset_bits;; offset = (offset - 1u) & table.offset_bits) {
				// The V# the shader would read, from guest memory that the GPU did not write.
				const auto address = entries + offset;
				if (!m_mapped_ranges.Contains(address, 12) ||
				    m_buffer_cache.IsRegionGpuModified(address, 12)) {
					return false;
				}
				ShaderBufferResource descriptor;
				std::memcpy(descriptor.fields, reinterpret_cast<const void*>(address), 12);
				const auto stride = uint64_t {descriptor.Stride()};
				if (descriptor.Base48() != 0 && descriptor.NumRecords() != 0) {
					const auto wanted =
					    descriptor.GetSize() + std::max<uint64_t>(stride, 4u) * WaveRecords;
					if (wanted > MaxBufferBytes) {
						Common::HackNotice::Report("DMA upload of a V# range clipped to 256 MiB",
						                           "RenderContext::SynchronizeDmaFootprint",
						                           program.shader_hash, table.base_register);
					}
					add(descriptor.Base48(), std::min(wanted, MaxBufferBytes));
				}
				if (offset == 0) {
					break;
				}
			}
		}
	}
	lock.unlock();
	m_buffer_cache.SynchronizeBuffersInSpans(m_dma_spans);
	return true;
}

void RenderContext::PrepareBda(bool may_write, bool full_sync) {
	m_buffer_cache.PrepareFaultBuffer();
	if (may_write) {
		m_buffer_cache.NoteDeviceAddressUse();
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	// Nothing to upload when no page became CPU dirty, no buffer was registered and no range was
	// mapped since the last synchronization started.
	const std::array<uint64_t, 3> epochs {RegionManager::CpuDirtyEpoch(),
	                                      m_buffer_cache.RegisterEpoch(), m_mapped_ranges_version};
	// The guest's CPU writes that the work of a submission may read through device addresses
	// precede the submission, or a WAIT_REG_MEM that the work waited for; the GPU sees later ones
	// only after such a wait. So once a synchronization ran for a submission, pages the CPU
	// dirties afterwards wait for the next submission or wait (they stay dirty meanwhile),
	// unless the execution thread itself wrote them for the command stream (labels, CP writes)
	// or the buffers or mappings changed.
	const bool same_layout = epochs[1] == m_bda_sync_epochs[1] && epochs[2] == m_bda_sync_epochs[2];
	if (full_sync && epochs != m_bda_sync_epochs &&
	    (!same_layout || m_bda_synced_generation != m_bda_generation)) {
		m_bda_synced_generation = m_bda_generation;
		// A new buffer or mapped range may cover CPU-dirty pages that earlier passes skipped: the
		// buffer cache forces the regions of new buffers, and of the ranges mapped since. Only this
		// thread takes the ranges out; MapMemory() adds them under the exclusive lock.
		for (const auto& [vaddr, size]: m_bda_mapped_since_sync) {
			m_buffer_cache.ForceSynchronization(vaddr, size);
		}
		m_bda_mapped_since_sync.clear();
		m_buffer_cache.SynchronizeBuffersInRanges(m_mapped_ranges, false);
		m_bda_sync_epochs = epochs;
	}
	m_fault_process_pending = true;
	m_use_scan_pending      = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	constexpr uint32_t UseScanPeriod = 256;
	if (m_use_scan_countdown != 0) {
		m_use_scan_countdown--;
	} else if (m_use_scan_pending) {
		m_use_scan_pending   = false;
		m_use_scan_countdown = UseScanPeriod;
		m_buffer_cache.ProcessUseBuffer();
	}
	if (m_indirect_write_tables.TakePendingWrites()) {
		m_buffer_cache.ProcessWriteBuffer([this](std::span<const uint64_t> pages) {
			m_indirect_write_tables.NoteWrites(pages);
		});
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
