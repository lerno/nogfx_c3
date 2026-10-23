module nogfx @feat(!DARWIN, NOGFX_VULKAN);
import std::io, std::os, std::collections, libc;

// ---- Presentation bookkeeping ---------------------------------------------------------------------------------------------------
//
// A swapchain image is recycled in two steps. The present fence of its context says the presentation engine is done with the image; the
// presentation retirement timeline, signaled by every presenting submission, says the GPU is done with the commands that used it. A
// replaced swapchain is destroyed only after both, image view by image view, through the delete queue.

// Destroys the swapchain images that the presentation timeline has passed, and a swapchain with its last image.
fn void Device.destroy_retired_images(&self)
{
	while (!self.retired_images.is_empty() && self.retired_images.first().retire_value <= self.completed_presentation_retirement)
	{
		DeferredSwapchainImage image = self.retired_images.pop_first()!!;
		self.vk.vkDestroyImageView(self.device, image.view, null);
		if (self.retired_images.is_empty() || self.retired_images.first().swapchain != image.swapchain) self.vk.vkDestroySwapchainKHR(self.device, image.swapchain, null);
	}
}

fn void Device.poll_presentation_retirement(&self)
{
	if (!self.presentation_retirement || self.completed_presentation_retirement == self.presentation_retirement_value) return;
	ulong completed;
	assert_vk(self.vk.vkGetSemaphoreCounterValue(self.device, self.presentation_retirement, &completed));
	assert(completed >= self.completed_presentation_retirement && completed <= self.presentation_retirement_value);
	if (completed == self.completed_presentation_retirement) return;
	self.completed_presentation_retirement = completed;
	self.destroy_retired_images();
}

fn void Device.wait_presentation_retirement(&self, ulong value)
{
	assert(value <= self.presentation_retirement_value);
	if (value <= self.completed_presentation_retirement)
	{
		self.destroy_retired_images();
		return;
	}
	VkSemaphoreWaitInfo wait_info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = 1,
		.pSemaphores = &self.presentation_retirement,
		.pValues = &value,
	};
	assert_vk(self.vk.vkWaitSemaphores(self.device, &wait_info, ulong::max));
	self.completed_presentation_retirement = value;
	self.destroy_retired_images();
}

fn ulong Device.next_presentation_retirement(&self)
{
	ulong next = self.presentation_retirement_value + 1;
	if (next - self.completed_presentation_retirement > self.max_timeline_value_difference)
	{
		self.poll_presentation_retirement();
		if (next - self.completed_presentation_retirement > self.max_timeline_value_difference)
		{
			self.wait_presentation_retirement(next - self.max_timeline_value_difference);
		}
		assert(next - self.completed_presentation_retirement <= self.max_timeline_value_difference);
	}
	self.presentation_retirement_value = next;
	return next;
}

fn void Device.destroy_present_context(&self, PresentContext* context)
{
	assert(!context.present_pending);
	if (self.device)
	{
		if (context.acquired) self.vk.vkDestroySemaphore(self.device, context.acquired, null);
		if (context.rendered) self.vk.vkDestroySemaphore(self.device, context.rendered, null);
		if (context.presented) self.vk.vkDestroyFence(self.device, context.presented, null);
	}
	*context = {};
}

fn void? Device.create_present_context(&self, PresentContext* context)
{
	assert(!context.acquired && !context.rendered && !context.presented && !context.swapchain && !context.present_pending);
	VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	defer catch self.destroy_present_context(context);
	check_vk(self.vk.vkCreateSemaphore(self.device, &semaphore_info, null, &context.acquired), "vkCreateSemaphore (acquire)")!;
	check_vk(self.vk.vkCreateSemaphore(self.device, &semaphore_info, null, &context.rendered), "vkCreateSemaphore (render)")!;
	VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	check_vk(self.vk.vkCreateFence(self.device, &fence_info, null, &context.presented), "vkCreateFence")!;
}

fn void Device.queue_retired_swapchain(&self, RetiredSwapchain* retired)
{
	assert(retired.handle && retired.view_count != 0);
	for (uint i = 0; i < retired.view_count; i++)
	{
		assert(retired.views[i]);
		self.retired_images.push({ .retire_value = self.presentation_retirement_value, .swapchain = retired.handle, .view = retired.views[i] });
	}
	*retired = {};
	self.destroy_retired_images();
}

// Whether an image of the swapchain is still being presented.
fn bool Device.is_presenting(&self, VkSwapchainKHR swapchain)
{
	foreach (&context : self.present_contexts[:self.present_context_count])
	{
		if (context.present_pending && context.swapchain == swapchain) return true;
	}
	return false;
}

fn void Device.finish_present_context(&self, PresentContext* context)
{
	assert(!context.present_pending && context.swapchain);
	VkSwapchainKHR completed_swapchain = context.swapchain;
	context.swapchain = {};
	foreach (&retired : self.retired_swapchains)
	{
		if (retired.handle == completed_swapchain && !self.is_presenting(completed_swapchain)) self.queue_retired_swapchain(retired);
	}
}

fn void Device.wait_present_context(&self, PresentContext* context)
{
	if (!context.present_pending) return;
	assert_vk(self.vk.vkWaitForFences(self.device, 1, &context.presented, VK_TRUE, ulong::max));
	context.present_pending = false;
	self.finish_present_context(context);
}

fn void Device.poll_present_contexts(&self)
{
	for (uint i = 0; i < self.present_context_count; i++)
	{
		PresentContext* context = &self.present_contexts[i];
		if (!context.present_pending) continue;
		VkResult result = self.vk.vkGetFenceStatus(self.device, context.presented);
		if (result == VK_NOT_READY) continue;
		assert_vk(result);
		context.present_pending = false;
		self.finish_present_context(context);
	}
}

fn void Device.drain_contexts(&self)
{
	self.wait_presentation_retirement(self.presentation_retirement_value);
	for (uint i = 0; i < self.present_context_count; i++) self.wait_present_context(&self.present_contexts[i]);
	foreach (&retired : self.retired_swapchains) assert(!retired.handle);
	self.destroy_retired_images();
}

// ---- Choosing a device ----------------------------------------------------------------------------------------------------------

struct QueriedFeatures
{
	VkPhysicalDeviceFeatures2 core;
	VkPhysicalDeviceVulkan11Features vulkan11;
	VkPhysicalDeviceVulkan12Features vulkan12;
	VkPhysicalDeviceVulkan13Features vulkan13;
	VkPhysicalDeviceVulkan14Features vulkan14;
	VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptor_heap;
	VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR address_commands;
	VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_pointers;
	VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified_image_layouts;
	VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader;
	VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR swapchain_maintenance1;
}

<*
 Zeroes the features and links the chain. It holds pointers into itself: initialize it where it stays.
*>
fn void QueriedFeatures.init(&self, bool presentation, bool include_unified_image_layouts, bool include_address_commands)
{
	*self = {};
	self.core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	self.vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	self.vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	self.vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	self.vulkan14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
	self.descriptor_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
	self.address_commands.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR;
	self.untyped_pointers.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR;
	self.unified_image_layouts.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR;
	self.mesh_shader.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
	self.swapchain_maintenance1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR;
	self.core.pNext = &self.vulkan11;
	self.vulkan11.pNext = &self.vulkan12;
	self.vulkan12.pNext = &self.vulkan13;
	self.vulkan13.pNext = &self.vulkan14;
	self.vulkan14.pNext = &self.descriptor_heap;
	self.descriptor_heap.pNext = include_address_commands ? (void*)&self.address_commands : (void*)&self.untyped_pointers;
	self.address_commands.pNext = &self.untyped_pointers;
	self.untyped_pointers.pNext = include_unified_image_layouts ? (void*)&self.unified_image_layouts : (void*)&self.mesh_shader;
	self.unified_image_layouts.pNext = &self.mesh_shader;
	self.mesh_shader.pNext = presentation ? (void*)&self.swapchain_maintenance1 : null;
}

struct Candidate
{
	VkPhysicalDevice physical_device;
	uint[QUEUE_TYPE_COUNT] queue_families;
	uint[QUEUE_TYPE_COUNT] queue_counts;
	bool[QUEUE_TYPE_COUNT] timestamps;
	Uint3 copy_texture_granularity;
	VkPhysicalDeviceProperties properties;
	VkPhysicalDeviceMemoryProperties memory_properties;
	bool unified_image_layouts;
	bool address_commands;
	bool image_cube_array;
	bool texture_compression_bc;
	bool texture_compression_astc;
	bool texture_compression_etc2;
	bool storage_input_output16;
	bool khr_swapchain_maintenance1;
	VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_properties;
	VkPhysicalDeviceVulkan12Properties vulkan12_properties;
	FixedList{ZString, 5} missing_extensions;   // Why it is not this one, when it lacks extensions.
}

fn void append_capability_diagnostic(DString* message, String subject, ZString[] missing_extensions) @private
{
	(void)message.appendf("\n%s: %s", subject, missing_extensions.len ? "missing required Vulkan extensions:" : "required Vulkan 1.4, memory, shader, or presentation capabilities are unavailable.");
	foreach (name : missing_extensions) (void)message.appendf("\n  %s", name);
}

<*
 Whether the physical device can run the backend, and what it offers. UNSUPPORTED means "not this one", and `output` says which
 extensions it lacks. VK_KHR_device_address_commands is optional: without it addresses resolve to buffers.
*>
fn void? Device.inspect_candidate(&self, VkPhysicalDevice physical_device, SurfaceMaintenance maintenance, DeviceDesc* desc, Candidate* output)
{
	VkFns* vk = &self.vk;
	VkExtensionProperties[] extension_storage = mem::new_array(VkExtensionProperties, MAX_DEVICE_EXTENSIONS)[:MAX_DEVICE_EXTENSIONS];
	defer mem::free(extension_storage.ptr);
	uint extension_count = MAX_DEVICE_EXTENSIONS;
	VkResult enumerated = vk.vkEnumerateDeviceExtensionProperties(physical_device, null, &extension_count, extension_storage.ptr);
	if (enumerated == VK_INCOMPLETE) return UNSUPPORTED~;
	check_vk(enumerated, "vkEnumerateDeviceExtensionProperties")!;
	VkExtensionProperties[] extensions = extension_storage[:extension_count];

	bool unified_image_layouts_extension = has_extension(extensions, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
	ZString[4] required_extensions = {
		VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME,
		VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME,
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,   // Only with a surface.
	};
	foreach (name : required_extensions[:self.surface ? 4 : 3])
	{
		if (!has_extension(extensions, name)) output.missing_extensions.push(name);
	}
	bool khr_swapchain_maintenance1 = maintenance.khr && has_extension(extensions, VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
	bool ext_swapchain_maintenance1 = maintenance.ext && has_extension(extensions, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
	if (self.surface && !khr_swapchain_maintenance1 && !ext_swapchain_maintenance1)
	{
		output.missing_extensions.push(maintenance.khr && maintenance.ext
			? "VK_KHR_swapchain_maintenance1 or VK_EXT_swapchain_maintenance1"
			: maintenance.khr ? VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME : VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
	}
	if (output.missing_extensions.size) return UNSUPPORTED~;

	Candidate result = { .physical_device = physical_device, .copy_texture_granularity = { 1, 1, 1 } };
	result.heap_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;
	result.vulkan12_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
	result.heap_properties.pNext = &result.vulkan12_properties;
	VkPhysicalDeviceProperties2 properties2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &result.heap_properties };
	vk.vkGetPhysicalDeviceProperties2(physical_device, &properties2);
	result.properties = properties2.properties;
	result.heap_properties.pNext = null;
	result.vulkan12_properties.pNext = null;
	if (result.properties.apiVersion < VK_API_VERSION_1_4) return UNSUPPORTED~;

	vk.vkGetPhysicalDeviceMemoryProperties(physical_device, &result.memory_properties);
	bool cpu_visible_memory;
	for (uint i = 0; i < result.memory_properties.memoryTypeCount; i++)
	{
		if (is_usable_memory_type(&result.memory_properties, i) && (result.memory_properties.memoryTypes[i].propertyFlags & CPU_VISIBLE_MEMORY) == CPU_VISIBLE_MEMORY)
		{
			cpu_visible_memory = true;
			break;
		}
	}
	if (!cpu_visible_memory) return UNSUPPORTED~;

	bool address_commands_extension = has_extension(extensions, VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME);
	QueriedFeatures features;
	features.init(self.surface != (VkSurfaceKHR)0, unified_image_layouts_extension, address_commands_extension);
	vk.vkGetPhysicalDeviceFeatures2(physical_device, &features.core);
	VkPhysicalDeviceFeatures* core = &features.core.features;
	bool required_features = core.shaderInt16 && core.samplerAnisotropy && core.depthBiasClamp && core.independentBlend
		&& core.fragmentStoresAndAtomics && core.vertexPipelineStoresAndAtomics && core.shaderStorageImageReadWithoutFormat
		&& core.shaderStorageImageWriteWithoutFormat && core.multiDrawIndirect && core.drawIndirectFirstInstance
		&& features.vulkan11.storageBuffer16BitAccess && features.vulkan11.storagePushConstant16 && features.vulkan11.shaderDrawParameters
		&& features.vulkan12.shaderFloat16 && features.vulkan12.scalarBlockLayout && features.vulkan12.bufferDeviceAddress
		&& features.vulkan12.timelineSemaphore && features.vulkan13.synchronization2 && features.vulkan13.dynamicRendering
		&& features.vulkan13.maintenance4 && features.vulkan14.maintenance5 && features.descriptor_heap.descriptorHeap
		&& features.untyped_pointers.shaderUntypedPointers && features.mesh_shader.meshShader && features.mesh_shader.taskShader
		&& (core.textureCompressionBC || core.textureCompressionASTC_LDR)
		&& (!self.surface || features.swapchain_maintenance1.swapchainMaintenance1);
	if (!required_features) return UNSUPPORTED~;
	result.unified_image_layouts = unified_image_layouts_extension && features.unified_image_layouts.unifiedImageLayouts != 0;
	result.address_commands = address_commands_extension && features.address_commands.deviceAddressCommands != 0;
	result.image_cube_array = core.imageCubeArray != 0;
	result.texture_compression_bc = core.textureCompressionBC != 0;
	result.texture_compression_astc = core.textureCompressionASTC_LDR != 0;
	result.texture_compression_etc2 = core.textureCompressionETC2 != 0;
	result.storage_input_output16 = features.vulkan11.storageInputOutput16 != 0;
	result.khr_swapchain_maintenance1 = khr_swapchain_maintenance1;

	uint available_queue_count;
	vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &available_queue_count, null);
	if (available_queue_count > MAX_QUEUE_FAMILIES) return UNSUPPORTED~;
	VkQueueFamilyProperties[MAX_QUEUE_FAMILIES] queues;
	uint queue_count = available_queue_count;
	vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_count, &queues[0]);
	for (uint index = 0; index < queue_count; index++)
	{
		if (queues[index].queueCount == 0) continue;
		VkQueueFlags flags = queues[index].queueFlags;
		uint type = 0;
		if (flags & VK_QUEUE_GRAPHICS_BIT)
		{
			if (!(flags & VK_QUEUE_COMPUTE_BIT)) continue;
		}
		else if (flags & VK_QUEUE_COMPUTE_BIT)
		{
			type = 1;
		}
		else if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & ~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_PROTECTED_BIT)))
		{
			type = 2;
		}
		else
		{
			continue;
		}
		if (result.queue_counts[type] != 0) continue;
		VkBool32 presentation_supported = VK_TRUE;
		if (self.surface && type == 0)
		{
			check_vk(vk.vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, index, self.surface, &presentation_supported), "vkGetPhysicalDeviceSurfaceSupportKHR")!;
		}
		if (presentation_supported == VK_TRUE)
		{
			result.queue_families[type] = index;
			result.queue_counts[type] = queues[index].queueCount;
			result.timestamps[type] = desc.timestamp_query_count != 0 && queues[index].timestampValidBits == 64 && features.vulkan12.hostQueryReset != 0;
			if (type == 2)
			{
				result.copy_texture_granularity = {
					queues[index].minImageTransferGranularity.width,
					queues[index].minImageTransferGranularity.height,
					queues[index].minImageTransferGranularity.depth,
				};
			}
		}
	}
	if (result.queue_counts[0] == 0 || (desc.desired_compute_queue_count && result.queue_counts[1] == 0)
		|| (desc.desired_copy_queue_count && result.queue_counts[2] == 0))
	{
		return UNSUPPORTED~;
	}
	*output = result;
}

fn VkBool32 debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types, VkDebugUtilsMessengerCallbackDataEXT* data, void* user) @private
{
	// The library does not depend on the application's logging. Applications can install a messenger of their own; this one makes
	// validation failures visible in a debugger.
	if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT && data && data.pMessage) io::eprintfn("NoGraphicsAPI validation: %s", data.pMessage);
	return VK_FALSE;
}

fn os::DylibHandle? open_loader() @private
{
	$switch:
		$case env::WIN32:
			return os::dlopen_any("vulkan-1.dll");
		$case env::DARWIN:
			return os::dlopen_any("libvulkan.dylib", "libvulkan.1.dylib", "/usr/local/lib/libvulkan.dylib", "/opt/homebrew/lib/libvulkan.dylib");
		$default:
			return os::dlopen_any("libvulkan.so.1", "libvulkan.so");
	$endswitch
}

// Reports and returns the fault, for `return fail(...)~`. Without a message the usual one for a device that cannot do it.
fn fault fail(fault reason, String message = "Required Vulkan device, memory, or display capabilities are unavailable.") @private
{
	report(message);
	return reason;
}

extern fn void* get_module_handle_w(void* name) @cname("GetModuleHandleW") @feat(WIN32);

// ---- Device ---------------------------------------------------------------------------------------------------------------------

<*
 Creates the device on the best GPU that has what the backend needs: Vulkan 1.4, VK_EXT_descriptor_heap, VK_KHR_shader_untyped_pointers,
 VK_EXT_mesh_shader, and the memory, shader and queue features of upstream's fixed baseline. VK_KHR_device_address_commands is used when
 there. A discrete GPU is preferred. The message of an UNSUPPORTED says, for each GPU, which extensions it lacks. A headless device has no window; with a window (an HWND, Windows only) the device presents to it.
 Faults: UNSUPPORTED (no Vulkan, no such GPU, or a window on a platform without presentation), DRIVER_ERROR, OUT_OF_MEMORY, DEVICE_LOST.
*>
fn Device*? create_device(DeviceDesc desc = {})
{
	desc.desired_queue_count = or_one(desc.desired_queue_count);
	desc.desired_swapchain_image_count = desc.desired_swapchain_image_count ?: 2;
	if (desc.swapchain_format == UNDEFINED) desc.swapchain_format = BGRA8_UNORM;
	assert(desc.desired_swapchain_image_count <= MAX_SWAPCHAIN_IMAGES, "swapchain image count must fit the presentation context array");
	Device* device = mem::new(Device);
	defer catch device.destroy();
	device.init(&desc)!;
	return device;
}

// Which of the two surface maintenance extensions the instance has. The device needs the swapchain extension that matches one of them.
struct SurfaceMaintenance
{
	bool khr;
	bool ext;
}

// Fills in a zeroed device. A failure leaves it half made, which `destroy` copes with.
fn void? Device.init(&self, DeviceDesc* desc)
{
	self.retired_images.init(mem);
	self.texture_heap_alignment = 16;
	self.texture_memory_type = VK_MAX_MEMORY_TYPES;
	self.heap_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;
	self.surface_provider = desc.surface_provider;
	self.presentation = desc.window != null || desc.surface_provider.create_surface != null;
	self.timestamp_query_count = desc.timestamp_query_count;
	self.present_context_count = self.presentation ? desc.desired_swapchain_image_count : 0;

	SurfaceMaintenance maintenance = self.create_instance(desc)!;
	Candidate selected = self.select_physical_device(maintenance, desc)!;
	self.create_logical_device(desc, &selected)!;
	if (!self.supports_gpu_heap_memory() || !self.select_texture_memory_type()) return fail(UNSUPPORTED)~;
	self.check_entry_points(selected.address_commands)!;
	self.fill_caps(&selected);
	if (self.presentation) self.create_presentation(desc)!;
}

// Opens the loader and creates the instance, the debug messenger when validation is on, and the surface of the window.
fn SurfaceMaintenance? Device.create_instance(&self, DeviceDesc* desc)
{
	bool presentation = desc.window != null || desc.surface_provider.create_surface != null;
	os::DylibHandle? loader = open_loader();
	if (catch loader) return fail(UNSUPPORTED, "The Vulkan loader was not found: install a Vulkan driver.")~;
	self.loader = loader;
	VkFns* vk = &self.vk;
	vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)os::dlsym(self.loader, "vkGetInstanceProcAddr");
	if (!vk.vkGetInstanceProcAddr) return fail(UNSUPPORTED, "The Vulkan loader does not export vkGetInstanceProcAddr.")~;
	vk.load_global();
	if (!vk.vkEnumerateInstanceVersion || !vk.vkEnumerateInstanceExtensionProperties || !vk.vkCreateInstance)
	{
		return fail(UNSUPPORTED, "Vulkan 1.1 or newer loader is required.")~;
	}
	uint loader_version;
	check_vk(vk.vkEnumerateInstanceVersion(&loader_version), "vkEnumerateInstanceVersion")!;
	if (loader_version < vk_make_api_version(0, 1, 1, 0)) return fail(UNSUPPORTED, "Vulkan 1.1 or newer loader is required.")~;

	VkExtensionProperties[] instance_extension_storage = mem::new_array(VkExtensionProperties, MAX_INSTANCE_EXTENSIONS)[:MAX_INSTANCE_EXTENSIONS];
	defer mem::free(instance_extension_storage.ptr);
	uint instance_extension_count = MAX_INSTANCE_EXTENSIONS;
	VkResult enumerated = vk.vkEnumerateInstanceExtensionProperties(null, &instance_extension_count, instance_extension_storage.ptr);
	if (enumerated == VK_INCOMPLETE) return fail(UNSUPPORTED)~;
	check_vk(enumerated, "vkEnumerateInstanceExtensionProperties")!;
	VkExtensionProperties[] instance_extensions = instance_extension_storage[:instance_extension_count];

	SurfaceMaintenance maintenance;
	if (presentation)
	{
		self.swapchain_color_space_enabled = has_extension(instance_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
		maintenance.khr = has_extension(instance_extensions, VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
		maintenance.ext = has_extension(instance_extensions, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
		FixedList{ZString, 8} missing;
		if (!has_extension(instance_extensions, VK_KHR_SURFACE_EXTENSION_NAME)) missing.push(VK_KHR_SURFACE_EXTENSION_NAME);
		if (!has_extension(instance_extensions, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)) missing.push(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
		$if env::WIN32:
			if (!desc.surface_provider.create_surface && !has_extension(instance_extensions, VK_KHR_WIN32_SURFACE_EXTENSION_NAME)) missing.push(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
		$endif
		if (desc.surface_provider.create_surface)
		{
			foreach (ext : desc.surface_provider.required_extensions)
			{
				if (!has_extension(instance_extensions, ext)) missing.push(ext);
			}
		}
		if (!maintenance.khr && !maintenance.ext) missing.push("VK_KHR_surface_maintenance1 or VK_EXT_surface_maintenance1");
		if (desc.swapchain_color_space != SRGB && !self.swapchain_color_space_enabled) missing.push(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
		if (missing.size)
		{
			DString message = dstring::temp("Cannot initialize Vulkan presentation.");
			append_capability_diagnostic(&message, "Vulkan instance", missing.array_view());
			return fail(UNSUPPORTED, message.str_view())~;
		}
	}

	FixedList{ZString, 16} instance_names;
	FixedList{ZString, 8} layer_names;
	bool debug_utils_available;
	$if env::COMPILER_SAFE_MODE:
		VkLayerProperties[MAX_INSTANCE_LAYERS] layer_storage;
		uint layer_count = MAX_INSTANCE_LAYERS;
		VkResult layers_enumerated = vk.vkEnumerateInstanceLayerProperties(&layer_count, &layer_storage[0]);
		if (layers_enumerated == VK_INCOMPLETE) return fail(UNSUPPORTED)~;
		check_vk(layers_enumerated, "vkEnumerateInstanceLayerProperties")!;
		debug_utils_available = has_extension(instance_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		if (debug_utils_available) instance_names.push(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		foreach (&layer : layer_storage[:layer_count])
		{
			if (((ZString)&layer.layerName[0]).str_view() == "VK_LAYER_KHRONOS_validation") layer_names.push("VK_LAYER_KHRONOS_validation");
		}
	$endif
	if (presentation)
	{
		if (self.swapchain_color_space_enabled) instance_names.push(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
		instance_names.push(VK_KHR_SURFACE_EXTENSION_NAME);
		instance_names.push(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
		$if env::WIN32:
			if (!desc.surface_provider.create_surface) instance_names.push(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
		$endif
		if (desc.surface_provider.create_surface)
		{
			foreach (ext : desc.surface_provider.required_extensions) instance_names.push(ext);
		}
		if (maintenance.khr) instance_names.push(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
		if (maintenance.ext) instance_names.push(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
	}

	VkApplicationInfo app_info = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "NoGraphicsAPI application",
		.applicationVersion = vk_make_api_version(0, 0, 1, 0),
		.pEngineName = "NoGraphicsAPI",
		.engineVersion = vk_make_api_version(0, 0, 1, 0),
		.apiVersion = VK_API_VERSION_1_4,
	};
	VkInstanceCreateInfo instance_info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app_info,
		.enabledLayerCount = (uint)layer_names.size,
		.ppEnabledLayerNames = &layer_names.entries[0],
		.enabledExtensionCount = (uint)instance_names.size,
		.ppEnabledExtensionNames = &instance_names.entries[0],
	};
	check_vk(vk.vkCreateInstance(&instance_info, null, &self.instance), "vkCreateInstance")!;
	vk.load_instance(self.instance);

	if (debug_utils_available)
	{
		if (!vk.vkCreateDebugUtilsMessengerEXT || !vk.vkDestroyDebugUtilsMessengerEXT) return fail(DRIVER_ERROR, "The driver did not provide the debug messenger entry points.")~;
		VkDebugUtilsMessengerCreateInfoEXT debug_info = {
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
			.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
				| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
			.pfnUserCallback = &debug_callback,
		};
		check_vk(vk.vkCreateDebugUtilsMessengerEXT(self.instance, &debug_info, null, &self.debug_messenger), "vkCreateDebugUtilsMessengerEXT")!;
	}

	if (desc.surface_provider.create_surface)
	{
		ulong? surf = desc.surface_provider.create_surface((void*)self.instance, desc.surface_provider.user_data);
		if (catch surf) return fail(DRIVER_ERROR, "surface creation callback failed")~;
		self.surface = (VkSurfaceKHR)surf;
	}
	else if (presentation)
	{
		$if env::WIN32:
			VkWin32SurfaceCreateInfoKHR surface_info = {
				.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
				.hinstance = get_module_handle_w(null),
				.hwnd = desc.window,
			};
			check_vk(vk.vkCreateWin32SurfaceKHR(self.instance, &surface_info, null, &self.surface), "vkCreateWin32SurfaceKHR")!;
		$else
			return fail(UNSUPPORTED, "Presentation requires desc.surface_provider on this platform.")~;
		$endif
	}
	return maintenance;
}

// The best GPU that has what the backend needs, a discrete one when there is one. Its properties become the device's.
fn Candidate? Device.select_physical_device(&self, SurfaceMaintenance maintenance, DeviceDesc* desc)
{
	VkFns* vk = &self.vk;
	VkPhysicalDevice[MAX_PHYSICAL_DEVICES] physical_devices;
	uint physical_device_count = MAX_PHYSICAL_DEVICES;
	VkResult enumerated = vk.vkEnumeratePhysicalDevices(self.instance, &physical_device_count, &physical_devices[0]);
	if (enumerated == VK_INCOMPLETE) return fail(UNSUPPORTED)~;
	check_vk(enumerated, "vkEnumeratePhysicalDevices")!;

	Candidate selected;
	bool has_selected;
	DString diagnostics = dstring::temp("No compatible GPU found.");
	foreach (physical_device : physical_devices[:physical_device_count])
	{
		Candidate candidate;
		if (catch err = self.inspect_candidate(physical_device, maintenance, desc, &candidate))
		{
			if (err != UNSUPPORTED) return err~;
			VkPhysicalDeviceProperties properties;
			vk.vkGetPhysicalDeviceProperties(physical_device, &properties);
			append_capability_diagnostic(&diagnostics, ((ZString)&properties.deviceName[0]).str_view(), candidate.missing_extensions.array_view());
			continue;
		}
		bool discrete = candidate.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
		if (!has_selected || discrete)
		{
			selected = candidate;
			has_selected = true;
		}
		if (discrete) break;
	}
	if (!has_selected) return fail(UNSUPPORTED, diagnostics.str_view())~;

	self.physical_device = selected.physical_device;
	self.physical_properties = selected.properties;
	self.heap_properties = selected.heap_properties;
	self.heap_properties.pNext = null;
	self.max_timeline_value_difference = selected.vulkan12_properties.maxTimelineSemaphoreValueDifference;
	self.memory_properties = selected.memory_properties;
	for (uint i = 1; i < FORMAT_COUNT; i++) self.format_features[i] = self.optimal_format_features((Format)i);
	return selected;
}

// Creates the logical device with the features of the baseline, the queues that the desc asks for and the GPU has, and fetches the queues.
fn void? Device.create_logical_device(&self, DeviceDesc* desc, Candidate* selected)
{
	VkFns* vk = &self.vk;
	bool presentation = self.presentation;
	QueriedFeatures enabled;
	enabled.init(presentation, selected.unified_image_layouts, selected.address_commands);
	self.texture_compression_etc2 = selected.texture_compression_etc2;
	enabled.core.features = {
		.imageCubeArray = (VkBool32)selected.image_cube_array, .samplerAnisotropy = VK_TRUE, .shaderInt16 = VK_TRUE, .depthBiasClamp = VK_TRUE,
		.independentBlend = VK_TRUE, .textureCompressionBC = (VkBool32)selected.texture_compression_bc,
		.textureCompressionASTC_LDR = (VkBool32)selected.texture_compression_astc, .textureCompressionETC2 = (VkBool32)selected.texture_compression_etc2,
		.fragmentStoresAndAtomics = VK_TRUE, .vertexPipelineStoresAndAtomics = VK_TRUE, .shaderStorageImageReadWithoutFormat = VK_TRUE,
		.shaderStorageImageWriteWithoutFormat = VK_TRUE, .multiDrawIndirect = VK_TRUE, .drawIndirectFirstInstance = VK_TRUE,
	};
	enabled.vulkan11.storageBuffer16BitAccess = enabled.vulkan11.storagePushConstant16 = enabled.vulkan11.shaderDrawParameters = VK_TRUE;
	enabled.vulkan11.storageInputOutput16 = (VkBool32)selected.storage_input_output16;
	enabled.vulkan12.shaderFloat16 = enabled.vulkan12.scalarBlockLayout = enabled.vulkan12.timelineSemaphore = enabled.vulkan12.bufferDeviceAddress = VK_TRUE;
	enabled.vulkan12.hostQueryReset = (VkBool32)(selected.timestamps[0] || selected.timestamps[1] || selected.timestamps[2]);
	enabled.vulkan13.synchronization2 = enabled.vulkan13.dynamicRendering = enabled.vulkan13.maintenance4 = VK_TRUE;
	enabled.vulkan14.maintenance5 = enabled.descriptor_heap.descriptorHeap = enabled.untyped_pointers.shaderUntypedPointers = VK_TRUE;
	enabled.mesh_shader.taskShader = enabled.mesh_shader.meshShader = enabled.swapchain_maintenance1.swapchainMaintenance1 = VK_TRUE;
	enabled.address_commands.deviceAddressCommands = (VkBool32)selected.address_commands;
	enabled.unified_image_layouts.unifiedImageLayouts = (VkBool32)selected.unified_image_layouts;

	uint[QUEUE_TYPE_COUNT] requested_counts = { desc.desired_queue_count, desc.desired_compute_queue_count, desc.desired_copy_queue_count };
	for (uint type = 0; type < QUEUE_TYPE_COUNT; type++)
	{
		self.queue_type_counts[type] = min(requested_counts[type], selected.queue_counts[type]);
		self.queue_count += self.queue_type_counts[type];
	}
	self.queues = mem::new_array(Queue, self.queue_count);
	float[] queue_priorities = mem::new_array(float, self.queue_count)[:self.queue_count];
	defer mem::free(queue_priorities.ptr);
	foreach (&priority : queue_priorities) *priority = 1.0f;
	VkDeviceQueueCreateInfo[QUEUE_TYPE_COUNT] queue_infos;
	uint first_queue;
	for (uint type = 0; type < QUEUE_TYPE_COUNT; type++)
	{
		if (self.queue_type_counts[type] == 0) continue;
		self.queue_families[self.queue_family_count] = selected.queue_families[type];
		queue_infos[self.queue_family_count++] = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = selected.queue_families[type],
			.queueCount = self.queue_type_counts[type],
			.pQueuePriorities = &queue_priorities[first_queue],
		};
		first_queue += self.queue_type_counts[type];
	}
	FixedList{ZString, 16} device_names;
	device_names.push_all({ VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME });
	if (selected.address_commands) device_names.push(VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME);
	if (selected.unified_image_layouts) device_names.push(VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
	if (presentation)
	{
		device_names.push(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		device_names.push(selected.khr_swapchain_maintenance1 ? VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME : VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
	}
	VkDeviceCreateInfo device_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &enabled.core,
		.queueCreateInfoCount = self.queue_family_count,
		.pQueueCreateInfos = &queue_infos[0],
		.enabledExtensionCount = (uint)device_names.size,
		.ppEnabledExtensionNames = &device_names.entries[0],
	};
	check_vk(vk.vkCreateDevice(self.physical_device, &device_info, null, &self.device), "vkCreateDevice")!;
	vk.load_device(self.device, vk.vkGetDeviceProcAddr);
	first_queue = 0;
	for (uint type = 0; type < QUEUE_TYPE_COUNT; type++)
	{
		for (uint i = 0; i < self.queue_type_counts[type]; i++)
		{
			Queue* queue = &self.queues[first_queue++];
			queue.command_infos.init(mem);
			queue.wait_infos.init(mem);
			queue.family_index = selected.queue_families[type];
			queue.timestamps = selected.timestamps[type];
			vk.vkGetDeviceQueue(self.device, queue.family_index, i, &queue.queue);
		}
	}
}

// The drivers hand out the commands of the extensions they enable, but not always all of them.
fn void? Device.check_entry_points(&self, bool address_commands)
{
	VkFns* vk = &self.vk;
	if (!vk.vkWriteSamplerDescriptorsEXT || !vk.vkWriteResourceDescriptorsEXT || !vk.vkCmdBindSamplerHeapEXT || !vk.vkCmdBindResourceHeapEXT
		|| !vk.vkCmdPushDataEXT || !vk.vkCmdDrawMeshTasksEXT)
	{
		return fail(DRIVER_ERROR, "The graphics driver did not provide the required Vulkan command entry points.")~;
	}
	self.buffer_map = mem::new(BufferAddressMap);
	if (address_commands)
	{
		if (!vk.vkCmdBindIndexBuffer3KHR || !vk.vkCmdDrawIndirect2KHR || !vk.vkCmdDrawIndexedIndirect2KHR || !vk.vkCmdDispatchIndirect2KHR
			|| !vk.vkCmdDrawMeshTasksIndirect2EXT || !vk.vkCmdCopyMemoryKHR || !vk.vkCmdCopyMemoryToImageKHR || !vk.vkCmdCopyImageToMemoryKHR)
		{
			return fail(DRIVER_ERROR, "The graphics driver did not provide the enabled Vulkan device address commands.")~;
		}
		return;
	}
	if (!vk.vkCmdDrawMeshTasksIndirectEXT) return fail(DRIVER_ERROR, "The graphics driver did not provide vkCmdDrawMeshTasksIndirectEXT.")~;
}

fn void Device.fill_caps(&self, Candidate* selected)
{
	ZString device_name = (ZString)&self.physical_properties.deviceName[0];
	usz name_length = min((usz)device_name.len(), (usz)(self.name.len - 1));
	mem::copy(&self.name[0], device_name, (sz)name_length);
	self.caps = {
		.device_name = (String)self.name[:name_length],
		.queue_count = self.queue_count,
		.general_queue_count = self.queue_type_counts[0],
		.compute_queue_count = self.queue_type_counts[1],
		.copy_queue_count = self.queue_type_counts[2],
		.copy_texture_granularity = selected.copy_texture_granularity,
		.max_push_data_size = self.heap_properties.maxPushDataSize,
		.texture_heap_alignment = self.texture_heap_alignment,
		.timestamp_period_ns = selected.properties.limits.timestampPeriod,
		.sub_texel_precision_bits = selected.properties.limits.subTexelPrecisionBits,
		.texture_compression_bc = selected.texture_compression_bc,
		.texture_compression_astc = selected.texture_compression_astc,
		.storage_input_output16 = selected.storage_input_output16,
		.indirect_mesh_draw = true,
	};
}

// The synchronization of the swapchain images, and the swapchain of the window.
fn void? Device.create_presentation(&self, DeviceDesc* desc)
{
	VkSemaphoreTypeCreateInfo type_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE };
	VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type_info };
	check_vk(self.vk.vkCreateSemaphore(self.device, &semaphore_info, null, &self.presentation_retirement), "vkCreateSemaphore (presentation)")!;
	for (uint i = 0; i < self.present_context_count; i++) self.create_present_context(&self.present_contexts[i])!;
	self.swapchain = mem::new(Swapchain, {
		.device = self,
		.format = desc.swapchain_format,
		.color_space = desc.swapchain_color_space == EXTENDED_SRGB_LINEAR ? VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
	});
	recreate_swapchain(self.swapchain)!;
}

<*
 Waits for nothing: call `wait_idle` first, release every frame and destroy everything created from the device before it.
*>
fn void Device.destroy(&self)
{
	assert(!self.acquired_swapchain, "device destroyed while a swapchain image is acquired");
	VkFns* vk = &self.vk;
	if (self.device)
	{
		self.drain_contexts();
		destroy_owned_swapchain(self);
		self.destroy_retired_images();
		assert(self.retired_images.is_empty());
		if (self.presentation_retirement) vk.vkDestroySemaphore(self.device, self.presentation_retirement, null);
	}
	foreach (&queue : self.queues[:self.queue_count])
	{
		queue.command_infos.free();
		queue.wait_infos.free();
	}
	mem::free(self.queues);
	mem::free(self.buffer_map);
	self.retired_images.free();
	for (uint i = 0; i < self.present_context_count; i++) self.destroy_present_context(&self.present_contexts[i]);
	if (self.device) vk.vkDestroyDevice(self.device, null);
	if (self.instance && self.surface) vk.vkDestroySurfaceKHR(self.instance, self.surface, null);
	if (self.instance && self.debug_messenger && vk.vkDestroyDebugUtilsMessengerEXT) vk.vkDestroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);
	if (self.instance) vk.vkDestroyInstance(self.instance, null);
	if (self.loader) self.loader.close();
	mem::free(self);
}

fn DeviceCaps* Device.get_caps(&self) => &self.caps;

fn bool Device.supports_texture_format(&self, Format format, TextureUsage usage)
{
	assert(format != UNDEFINED && (uint)usage != 0);
	TextureFormatInfo info = get_texture_format_info(format);
	if ((info.depth || info.stencil) && usage.color_attachment) return false;
	if (!info.depth && !info.stencil && usage.depth_stencil_attachment) return false;
	if (info.depth && info.stencil && (usage.transfer_source || usage.transfer_destination)) return false;
	switch (texture_compression(format))
	{
		case NONE: break;
		case ETC2:
			if (!self.texture_compression_etc2) return false;
		case ASTC:
			if (!self.caps.texture_compression_astc) return false;
		case BC:
			if (!self.caps.texture_compression_bc) return false;
	}
	VkFormatFeatureFlags2 required = required_format_features(usage);
	return (self.format_features[format.ordinal] & required) == required;
}

// ---- Timeline semaphores --------------------------------------------------------------------------------------------------------

fn TimelineSemaphore*? Device.create_timeline_semaphore(&self, ulong initial_value = 0)
{
	VkSemaphoreTypeCreateInfo timeline_info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
		.initialValue = initial_value,
	};
	VkSemaphoreCreateInfo create_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timeline_info };
	TimelineSemaphore* result = mem::new(TimelineSemaphore, { .device = self });
	defer catch mem::free(result);
	check_vk(self.vk.vkCreateSemaphore(self.device, &create_info, null, &result.semaphore), "vkCreateSemaphore (timeline)")!;
	return result;
}

fn void TimelineSemaphore.destroy(&self)
{
	self.device.vk.vkDestroySemaphore(self.device.device, self.semaphore, null);
	mem::free(self);
}

fn ulong TimelineSemaphore.completed_value(&self)
{
	ulong value;
	assert_vk(self.device.vk.vkGetSemaphoreCounterValue(self.device.device, self.semaphore, &value));
	return value;
}

fn void TimelinePoint.wait(self)
{
	assert(self.semaphore && self.semaphore.semaphore, "wait requires a live timeline semaphore");
	VkSemaphore handle = self.semaphore.semaphore;
	VkSemaphoreWaitInfo wait_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &handle, .pValues = &self.value };
	assert_vk(self.semaphore.device.vk.vkWaitSemaphores(self.semaphore.device.device, &wait_info, ulong::max));
}

<*
 Waits for every queue and every presentation to finish. The whole device drains; call it before destroying resources.
*>
fn void Device.wait_idle(&self)
{
	assert(!self.acquired_swapchain, "wait_idle is not allowed while a swapchain image is acquired");
	assert_vk(self.vk.vkDeviceWaitIdle(self.device));
	self.drain_contexts();
	self.next_present_context = 0;
}

// ---- GPU heaps ------------------------------------------------------------------------------------------------------------------

<*
 Allocates a heap of raw memory and returns its CPU and GPU addresses. Every non null pointer is 16 byte aligned. GPU_ONLY has no CPU address.
 Without VK_KHR_device_address_commands at most 64 heaps can be live per device, and the range of a copy must lie inside one heap.
 Faults: OUT_OF_MEMORY when no compatible memory type is large enough.
*>
fn GpuHeap? Device.create_gpu_heap(&self, ulong byte_count, MemoryType memory = CPU_VISIBLE) => self.allocate_gpu_heap(byte_count, memory);

fn void GpuHeap.destroy(&self)
{
	if (!self.owner) return;
	Device* device = self.owner.device;
	BackingBuffer* backing = &self.owner.backing;
	if (self.owner.record) device.buffer_map.remove(self.owner.record);
	if (backing.mapped) device.vk.vkUnmapMemory(device.device, backing.memory);
	device.vk.vkDestroyBuffer(device.device, backing.buffer, null);
	device.vk.vkFreeMemory(device.device, backing.memory, null);
	mem::free(self.owner);
}

fn GpuRange GpuHeap.range(&self) => { self.gpu, self.size };

// ---- Texture heaps --------------------------------------------------------------------------------------------------------------

fn TextureHeap? Device.create_texture_heap(&self, ulong byte_count)
{
	TextureHeapOwner* owner = mem::new(TextureHeapOwner, { .device = self });
	defer catch mem::free(owner);
	VkMemoryAllocateInfo allocate_info = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = byte_count, .memoryTypeIndex = self.texture_memory_type };
	check_vk(self.vk.vkAllocateMemory(self.device, &allocate_info, null, &owner.memory), "vkAllocateMemory (texture heap)", byte_count)!;
	return { .size = byte_count, .owner = owner };
}

fn void TextureHeap.destroy(&self)
{
	if (!self.owner) return;
	self.owner.device.vk.vkFreeMemory(self.owner.device.device, self.owner.memory, null);
	mem::free(self.owner);
}
