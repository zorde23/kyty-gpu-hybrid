// cs_replay: runs one compute dispatch of a translated shader outside the game, with buffers and push
// data recorded by KYTY_HYB_TRACE_CS / KYTY_HYB_TRACE_CS_DUMP, and checks a radix sort pass against a
// CPU reference. No game data lives here: every input comes from the command line.
//
// cs_replay <shader.spv> <dump folder> <dispatch> <groups x> <wave size> <push words in hex...>
//   [--pass P] [--input I] [--output O] [--timeout-ms T]
// Buffer i of the dispatch is read from <dump folder>/dispatch<dispatch>_buffer<i>_*.bin and bound as
// element i of the storage buffer array at set 0, binding 0. With --pass, buffer I holds 16-byte
// elements (64-bit key, 64-bit value) and buffer O must end as I stably sorted by key byte P.
// "synthetic:<count>" in place of the dump folder runs lookback.comp on generated values instead and
// checks their inclusive prefix sum (the lookback_subgroup_fences test).
//
// Built by the cs_replay target when CMake finds the Vulkan loader (VULKAN_SDK); exit code 77 means no device.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

[[noreturn]] void Fail(const char* what, VkResult result = VK_SUCCESS) {
	std::fprintf(stderr, "cs_replay: %s (VkResult %d)\n", what, static_cast<int>(result));
	std::exit(2);
}

void Check(VkResult result, const char* what) {
	if (result != VK_SUCCESS) Fail(what, result);
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) Fail(("cannot read " + path.string()).c_str());
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::filesystem::path FindDump(const std::filesystem::path& folder, uint32_t dispatch, uint32_t buffer) {
	const auto prefix = "dispatch" + std::to_string(dispatch) + "_buffer" + std::to_string(buffer) + "_";
	for (const auto& entry: std::filesystem::directory_iterator(folder)) {
		if (entry.path().filename().string().rfind(prefix, 0) == 0) return entry.path();
	}
	return {};
}

struct Element {
	uint64_t key;
	uint64_t value;
	bool     operator==(const Element&) const = default;
	bool     operator<(const Element& other) const { return key != other.key ? key < other.key : value < other.value; }
};

std::vector<Element> Elements(const std::vector<uint8_t>& bytes, size_t count) {
	std::vector<Element> elements(count);
	std::memcpy(elements.data(), bytes.data(), count * sizeof(Element));
	return elements;
}

} // namespace

int main(int argc, char** argv) {
	if (argc < 7) {
		std::fprintf(stderr, "usage: cs_replay <shader.spv> <dump folder> <dispatch> <groups x> <wave size> <push words...>"
		                     " [--pass P] [--input I] [--output O] [--timeout-ms T] [--compile-only]\n");
		return 2;
	}
	const std::filesystem::path spv_path = argv[1];
	const std::filesystem::path folder   = argv[2];
	const auto                  dispatch = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
	const auto                  groups   = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
	const auto                  wave     = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
	std::vector<uint32_t>       push(32, 0);
	int                         pass = -1, input = 1, output = 4;
	uint64_t                    timeout_ms = 3000;
	size_t                      words      = 0;
	bool                        compile_only = false;
	for (int i = 6; i < argc; i++) {
		const std::string arg = argv[i];
		if (arg == "--compile-only") compile_only = true;
		else if (arg == "--pass" && i + 1 < argc) pass = std::atoi(argv[++i]);
		else if (arg == "--input" && i + 1 < argc) input = std::atoi(argv[++i]);
		else if (arg == "--output" && i + 1 < argc) output = std::atoi(argv[++i]);
		else if (arg == "--timeout-ms" && i + 1 < argc) timeout_ms = std::strtoull(argv[++i], nullptr, 10);
		else if (words < push.size()) push[words++] = static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 16));
	}

	// The buffers of the dispatch, in descriptor order. "synthetic:<count>" instead of a dump folder makes
	// the three buffers of lookback.comp (ticket, tile status, values) with pseudo-random values, and the
	// check becomes an inclusive prefix sum of the values.
	std::vector<std::vector<uint8_t>> buffers;
	std::vector<uint32_t>             synthetic;
	const auto                        folder_text = folder.string();
	if (folder_text.rfind("synthetic:", 0) == 0) {
		synthetic.resize(std::strtoul(folder_text.c_str() + 10, nullptr, 10));
		uint32_t seed = 12345;
		for (auto& value: synthetic) value = (seed = seed * 1664525u + 1013904223u) >> 28;
		buffers.push_back(std::vector<uint8_t>(4, 0));
		buffers.push_back(std::vector<uint8_t>(groups * 4, 0));
		buffers.push_back(std::vector<uint8_t>(synthetic.size() * 4));
		std::memcpy(buffers[2].data(), synthetic.data(), buffers[2].size());
		push[0] = static_cast<uint32_t>(synthetic.size());
	} else {
		for (uint32_t i = 0;; i++) {
			const auto path = FindDump(folder, dispatch, i);
			if (path.empty()) break;
			buffers.push_back(ReadFile(path));
		}
	}
	if (buffers.empty()) Fail("no buffer dumps for that dispatch");
	const auto spv = ReadFile(spv_path);

	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	VkInstance instance;
	Check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> physicals(count);
	vkEnumeratePhysicalDevices(instance, &count, physicals.data());
	if (physicals.empty()) {
		std::fprintf(stderr, "cs_replay: no Vulkan device, skipped\n");
		return 77;
	}
	VkPhysicalDevice physical = physicals[0];
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(physical, &properties);
	std::printf("device: %s\n", properties.deviceName);

	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
	std::vector<VkQueueFamilyProperties> families(count);
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
	uint32_t family = 0;
	while (family < count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) family++;

	// A required subgroup size (the guest's wave size) needs subgroupSizeControl; 0 leaves the driver's choice.
	VkPhysicalDeviceVulkan13Features features13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	features13.subgroupSizeControl = wave != 0 ? VK_TRUE : VK_FALSE;
	VkPhysicalDeviceFeatures2 features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	features.pNext                 = &features13;
	features.features.shaderInt64 = VK_TRUE;
	const float             priority = 1.0f;
	VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.pNext                = &features;
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos    = &queue_info;
	VkDevice device;
	Check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
	VkQueue queue;
	vkGetDeviceQueue(device, family, 0, &queue);

	VkPhysicalDeviceMemoryProperties memory_properties;
	vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
	auto host_memory_type = [&](uint32_t bits) {
		const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
			if ((bits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
		}
		Fail("no host-visible memory type");
	};

	struct Buffer {
		VkBuffer       buffer;
		VkDeviceMemory memory;
		void*          mapped;
		VkDeviceSize   size;
	};
	std::vector<Buffer> gpu(buffers.size());
	for (size_t i = 0; i < buffers.size(); i++) {
		auto&              b = gpu[i];
		VkBufferCreateInfo buffer_info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		b.size            = std::max<VkDeviceSize>(buffers[i].size(), 4);
		buffer_info.size  = b.size;
		buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		Check(vkCreateBuffer(device, &buffer_info, nullptr, &b.buffer), "vkCreateBuffer");
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, b.buffer, &requirements);
		VkMemoryAllocateInfo allocate {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		allocate.allocationSize  = requirements.size;
		allocate.memoryTypeIndex = host_memory_type(requirements.memoryTypeBits);
		Check(vkAllocateMemory(device, &allocate, nullptr, &b.memory), "vkAllocateMemory");
		Check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
		Check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
		std::memcpy(b.mapped, buffers[i].data(), buffers[i].size());
	}

	VkDescriptorSetLayoutBinding binding {};
	binding.binding         = 0;
	binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	binding.descriptorCount = static_cast<uint32_t>(gpu.size());
	binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	VkDescriptorSetLayoutCreateInfo set_layout_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	set_layout_info.bindingCount = 1;
	set_layout_info.pBindings    = &binding;
	VkDescriptorSetLayout set_layout;
	Check(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout");
	VkPushConstantRange push_range {VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push.size() * 4)};
	VkPipelineLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_range;
	VkPipelineLayout layout;
	Check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout), "vkCreatePipelineLayout");

	VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module_info.codeSize = spv.size();
	module_info.pCode    = reinterpret_cast<const uint32_t*>(spv.data());
	VkShaderModule module;
	Check(vkCreateShaderModule(device, &module_info, nullptr, &module), "vkCreateShaderModule");
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {
	    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
	subgroup.requiredSubgroupSize = wave;
	VkComputePipelineCreateInfo pipeline_info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	pipeline_info.stage        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	pipeline_info.stage.pNext  = wave != 0 ? &subgroup : nullptr;
	pipeline_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = layout;
	VkPipeline pipeline;
	Check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines");
	if (compile_only) {
		std::printf("cs_replay: pipeline created\n");
		return 0;
	}

	VkDescriptorPoolSize       pool_size {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding.descriptorCount};
	VkDescriptorPoolCreateInfo pool_info {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	VkDescriptorPool pool;
	Check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool), "vkCreateDescriptorPool");
	VkDescriptorSetAllocateInfo set_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	set_info.descriptorPool     = pool;
	set_info.descriptorSetCount = 1;
	set_info.pSetLayouts        = &set_layout;
	VkDescriptorSet set;
	Check(vkAllocateDescriptorSets(device, &set_info, &set), "vkAllocateDescriptorSets");
	std::vector<VkDescriptorBufferInfo> infos;
	for (const auto& b: gpu) infos.push_back({b.buffer, 0, b.size});
	VkWriteDescriptorSet write {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
	write.dstSet          = set;
	write.descriptorCount = binding.descriptorCount;
	write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo     = infos.data();
	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

	VkCommandPoolCreateInfo command_pool_info {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	command_pool_info.queueFamilyIndex = family;
	VkCommandPool command_pool;
	Check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");
	VkCommandBufferAllocateInfo command_info {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	command_info.commandPool        = command_pool;
	command_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	command_info.commandBufferCount = 1;
	VkCommandBuffer command;
	Check(vkAllocateCommandBuffers(device, &command_info, &command), "vkAllocateCommandBuffers");
	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	vkBeginCommandBuffer(command, &begin);
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
	vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_range.size, push.data());
	vkCmdDispatch(command, groups, 1, 1);
	VkMemoryBarrier barrier {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
	                     nullptr, 0, nullptr);
	vkEndCommandBuffer(command);
	VkFenceCreateInfo fence_info {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence           fence;
	Check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");
	VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
	const VkResult waited = vkWaitForFences(device, 1, &fence, VK_TRUE, timeout_ms * 1000000ull);
	if (waited == VK_TIMEOUT) {
		std::printf("HANG: the dispatch did not finish in %llu ms\n", static_cast<unsigned long long>(timeout_ms));
		return 3;
	}
	Check(waited, "vkWaitForFences");
	std::printf("finished\n");
	if (const char* save = std::getenv("CS_REPLAY_SAVE"); save != nullptr) {
		for (size_t i = 0; i < gpu.size(); i++) {
			std::ofstream file(std::string(save) + "/after_buffer" + std::to_string(i) + ".bin", std::ios::binary);
			file.write(static_cast<const char*>(gpu[i].mapped), static_cast<std::streamsize>(buffers[i].size()));
		}
	}

	if (!synthetic.empty()) {
		const auto* out   = static_cast<const uint32_t*>(gpu[2].mapped);
		uint32_t    sum   = 0;
		size_t      wrong = 0;
		for (size_t i = 0; i < synthetic.size(); i++) wrong += out[i] != (sum += synthetic[i]);
		std::printf("prefix sum: %zu values, %zu wrong\n", synthetic.size(), wrong);
		return wrong == 0 ? 0 : 1;
	}
	if (pass >= 0) {
		const size_t elements = push[0];
		auto         in       = Elements(buffers[input], elements);
		auto         out      = Elements(std::vector<uint8_t>(static_cast<uint8_t*>(gpu[output].mapped),
		                                                      static_cast<uint8_t*>(gpu[output].mapped) + gpu[output].size),
		                                 elements);
		auto reference = in;
		std::stable_sort(reference.begin(), reference.end(), [pass](const Element& a, const Element& b) {
			return ((a.key >> (8 * pass)) & 0xff) < ((b.key >> (8 * pass)) & 0xff);
		});
		size_t wrong = 0;
		for (size_t i = 0; i < elements; i++) wrong += !(out[i] == reference[i]);
		std::map<Element, int> balance;
		for (const auto& e: in) balance[e]++;
		for (const auto& e: out) balance[e]--;
		size_t lost = 0, extra = 0;
		for (const auto& [e, n]: balance) (n > 0 ? lost : extra) += static_cast<size_t>(std::abs(n));
		std::printf("pass %d: %zu elements, %zu differ from the stable reference, %zu lost, %zu extra\n", pass, elements,
		            wrong, lost, extra);
		return wrong == 0 ? 0 : 1;
	}
	return 0;
}
