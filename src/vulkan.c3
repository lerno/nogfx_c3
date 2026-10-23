module nogfx @feat(!DARWIN, NOGFX_VULKAN);
import std::io, std::os, std::collections, libc;

// The Vulkan 1.4 backend, a function by function port of src/NoGraphicsAPI.cpp in NoGraphicsAPI (MIT, Sebastian Aaltonen). It is built
// on Windows and Linux, and on macOS with -D NOGFX_VULKAN; src/metal.c3 is the backend everywhere else.
//
// Vulkan 1.4 is not enough: the device needs VK_EXT_descriptor_heap, VK_KHR_shader_untyped_pointers and VK_EXT_mesh_shader, which today means
// recent NVIDIA, AMD and Mesa RADV drivers. MoltenVK does not have them. VK_KHR_device_address_commands is optional: without it index, indirect
// and copy commands take buffers, and GPU addresses are resolved to them through a table of at most 64 live GPU heaps (address_map.c3).
// Presentation is Win32 only (DeviceDesc.window is an HWND), as upstream; a headless device works anywhere. Nothing links to Vulkan: the loader is opened
// when the device is created, and the commands are looked up in it (src/vk.c3, made by tools/gen_vk.py).
//
// The API is not thread safe except where upstream says so: queues and command pools are externally synchronized, and resource
// creation, immutable queries, timeline waits and descriptor writes to disjoint slots can run concurrently. There are no internal locks.
//
// The root pointer of a draw or dispatch is pushed with vkCmdPushDataEXT. Shaders read it as the uniform buffer at set 0 binding 0, which the
// pipeline maps to that address. Textures and samplers are indices into the bound descriptor heaps.

// ---- Constants ------------------------------------------------------------------------------------------------------------------

const uint MAX_DEVICE_EXTENSIONS = 512;
const uint MAX_INSTANCE_EXTENSIONS = 256;
const uint MAX_INSTANCE_LAYERS = 64;
const uint MAX_PHYSICAL_DEVICES = 32;
const uint MAX_QUEUE_FAMILIES = 64;
const uint QUEUE_TYPE_COUNT = 3;   // General, compute only, copy only.
const uint MAX_COLOR_ATTACHMENTS = 8;
const uint MAX_SWAPCHAIN_IMAGES = 8;
const uint GPU_ALLOCATION_ALIGNMENT = 16;
const uint FORMAT_COUNT = Format::values.len;
const VkPresentModeKHR SWAPCHAIN_PRESENT_MODE = VK_PRESENT_MODE_FIFO_KHR;

const VkBufferUsageFlags UNIVERSAL_BUFFER_USAGE = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
	| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
const VkMemoryPropertyFlags CPU_VISIBLE_MEMORY = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
	| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
const VkMemoryPropertyFlags FORBIDDEN_MEMORY = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT
	| VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
const VkAddressCommandFlagsKHR ADDRESS_FLAGS = VK_ADDRESS_COMMAND_FULLY_BOUND_BIT_KHR;

enum DescriptorHeapType : char { TEXTURE, SAMPLER }

// ---- Errors ---------------------------------------------------------------------------------------------------------------------

fn String vk_reason(VkResult result) @private
{
	switch (result)
	{
		case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED (graphics initialization failed)";
		case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY (not enough system memory)";
		case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY (not enough GPU memory or CPU-visible GPU memory)";
		case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS (graphics resource limit reached)";
		case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST (graphics device lost)";
		case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED (GPU memory could not be mapped for CPU access)";
		case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT (required Vulkan layer unavailable)";
		case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT (required Vulkan extension unavailable)";
		case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT (required graphics feature unavailable)";
		case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER (incompatible graphics driver)";
		case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED (required texture or display format unsupported)";
		default: return "graphics driver error";
	}
}

fn fault fault_from_vk(VkResult result) @private
{
	switch (result)
	{
		case VK_ERROR_OUT_OF_HOST_MEMORY:
		case VK_ERROR_OUT_OF_DEVICE_MEMORY:
		case VK_ERROR_TOO_MANY_OBJECTS: return OUT_OF_MEMORY;
		case VK_ERROR_DEVICE_LOST: return DEVICE_LOST;
		case VK_ERROR_LAYER_NOT_PRESENT:
		case VK_ERROR_EXTENSION_NOT_PRESENT:
		case VK_ERROR_FEATURE_NOT_PRESENT:
		case VK_ERROR_INCOMPATIBLE_DRIVER:
		case VK_ERROR_FORMAT_NOT_SUPPORTED: return UNSUPPORTED;
		default: return DRIVER_ERROR;
	}
}

// Diagnostics go to stderr and to the callback, as in the Metal backend.
fn void report(String message, bool fatal = false) @private
{
	io::eprintfn("NoGraphicsAPI Vulkan: %s", message);
	if (error_callback) error_callback(message, fatal);
}

fn void report_vk(VkResult result, String operation, bool fatal, ulong byte_count = 0) @private
{
	String reason = vk_reason(result);
	if (byte_count)
	{
		report(string::tformat("%s failed: %s (Vulkan result %d, %.2f MiB requested).", operation, reason, (int)result,
			(double)byte_count / (1024.0 * 1024.0)), fatal);
		return;
	}
	report(string::tformat("%s failed: %s (Vulkan result %d).", operation, reason, (int)result), fatal);
}

<*
 Reports a failure and returns its fault. The creation functions clean up and pass the fault on.
*>
fn void? check_vk(VkResult result, String operation, ulong byte_count = 0) @private
{
	if (result == VK_SUCCESS) return;
	report_vk(result, operation, false, byte_count);
	return fault_from_vk(result)~;
}

<*
 A failure of something that can only fail when the device is lost or memory is gone: report it and stop, as upstream does.
*>
fn void abort_vk(VkResult result, String operation) @private @noreturn
{
	report_vk(result, operation, true);
	libc::abort();
}

fn void require_vk(VkResult result, String operation) @private
{
	if (result != VK_SUCCESS) abort_vk(result, operation);
}

fn void assert_vk(VkResult result) @private
{
	assert(result == VK_SUCCESS, "unexpected Vulkan failure");
}

// ---- Helpers --------------------------------------------------------------------------------------------------------------------

fn ulong divide_up(ulong value, ulong divisor) @private
{
	assert(divisor != 0);
	return value / divisor + (ulong)(value % divisor != 0 ? 1 : 0);
}

fn bool has_extension(VkExtensionProperties[] values, ZString name) @private
{
	foreach (&value : values)
	{
		if (libc::strcmp((ZString)&value.extensionName[0], name) == 0) return true;
	}
	return false;
}

fn VkFormat vk_format(Format format) @private
{
	switch (format)
	{
		case R8_SRGB: return VK_FORMAT_R8_SRGB;
		case RG8_SRGB: return VK_FORMAT_R8G8_SRGB;
		case RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
		case BGRA8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
		case RGBA4_UNORM: return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
		case R5G5B5A1_UNORM: return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
		case R5G6B5_UNORM: return VK_FORMAT_R5G6B5_UNORM_PACK16;
		case R8_UNORM: return VK_FORMAT_R8_UNORM;
		case RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
		case RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
		case BGRA8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
		case R16_UNORM: return VK_FORMAT_R16_UNORM;
		case RG16_UNORM: return VK_FORMAT_R16G16_UNORM;
		case RGBA16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
		case R8_UINT: return VK_FORMAT_R8_UINT;
		case RG8_UINT: return VK_FORMAT_R8G8_UINT;
		case RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
		case BGRA8_UINT: return VK_FORMAT_B8G8R8A8_UINT;
		case R16_UINT: return VK_FORMAT_R16_UINT;
		case RG16_UINT: return VK_FORMAT_R16G16_UINT;
		case RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
		case R32_UINT: return VK_FORMAT_R32_UINT;
		case RG32_UINT: return VK_FORMAT_R32G32_UINT;
		case RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
		case R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
		case RG16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
		case RGBA16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
		case R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
		case RG32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
		case RGBA32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
		case RGB10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
		case RG11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
		case D16_UNORM: return VK_FORMAT_D16_UNORM;
		case D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
		case D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
		case S8_UINT: return VK_FORMAT_S8_UINT;
		case D32_FLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
		case EAC_RG: return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
		case ASTC_4X4_SRGB: return VK_FORMAT_ASTC_4X4_SRGB_BLOCK;
		case ASTC_4X4_UNORM: return VK_FORMAT_ASTC_4X4_UNORM_BLOCK;
		case BC3_SRGB: return VK_FORMAT_BC3_SRGB_BLOCK;
		case BC3_UNORM: return VK_FORMAT_BC3_UNORM_BLOCK;
		case BC5_RG: return VK_FORMAT_BC5_UNORM_BLOCK;
		case BC6H_UFLOAT: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
		case BC6H_SFLOAT: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
		case BC7_SRGB: return VK_FORMAT_BC7_SRGB_BLOCK;
		case BC7_UNORM: return VK_FORMAT_BC7_UNORM_BLOCK;
		case UNDEFINED: return VK_FORMAT_UNDEFINED;
	}
}

fn bool compatible_view_formats(Format image_format, Format view_format) @private
{
	if (image_format == view_format) return true;
	TextureFormatInfo image = get_texture_format_info(image_format);
	TextureFormatInfo view = get_texture_format_info(view_format);
	if (image.depth || image.stencil || view.depth || view.stencil) return false;
	if (image.block_extent.x == 1 && view.block_extent.x == 1) return image.bytes_per_block != 0 && image.bytes_per_block == view.bytes_per_block;
	return (image_format == ASTC_4X4_UNORM && view_format == ASTC_4X4_SRGB) || (image_format == ASTC_4X4_SRGB && view_format == ASTC_4X4_UNORM)
		|| (image_format == BC3_UNORM && view_format == BC3_SRGB) || (image_format == BC3_SRGB && view_format == BC3_UNORM)
		|| (image_format == BC6H_UFLOAT && view_format == BC6H_SFLOAT) || (image_format == BC6H_SFLOAT && view_format == BC6H_UFLOAT)
		|| (image_format == BC7_UNORM && view_format == BC7_SRGB) || (image_format == BC7_SRGB && view_format == BC7_UNORM);
}

fn VkImageAspectFlags image_aspects(Format format) @private
{
	TextureFormatInfo info = get_texture_format_info(format);
	VkImageAspectFlags result;
	if (info.depth) result |= VK_IMAGE_ASPECT_DEPTH_BIT;
	if (info.stencil) result |= VK_IMAGE_ASPECT_STENCIL_BIT;
	return result ?: VK_IMAGE_ASPECT_COLOR_BIT;
}

fn VkImageType vk_image_type(TextureType type) @private
{
	switch (type)
	{
		case ONE_D: return VK_IMAGE_TYPE_1D;
		case THREE_D: return VK_IMAGE_TYPE_3D;
		case TWO_D:
		case CUBE:
		case TWO_D_ARRAY:
		case CUBE_ARRAY: return VK_IMAGE_TYPE_2D;
	}
}

fn VkImageViewType vk_view_type(TextureType type) @private
{
	switch (type)
	{
		case ONE_D: return VK_IMAGE_VIEW_TYPE_1D;
		case TWO_D: return VK_IMAGE_VIEW_TYPE_2D;
		case THREE_D: return VK_IMAGE_VIEW_TYPE_3D;
		case CUBE: return VK_IMAGE_VIEW_TYPE_CUBE;
		case TWO_D_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
		case CUBE_ARRAY: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
	}
}

fn VkFormatFeatureFlags2 required_format_features(TextureUsage usage) @private
{
	VkFormatFeatureFlags2 result;
	if (usage.sampled) result |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
	if (usage.storage)
	{
		result |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT
			| VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
	}
	if (usage.color_attachment) result |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT;
	if (usage.depth_stencil_attachment) result |= VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
	if (usage.transfer_source) result |= VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT;
	if (usage.transfer_destination) result |= VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
	return result;
}

enum TextureCompression : char { NONE, ETC2, ASTC, BC }

fn TextureCompression texture_compression(Format format) @private
{
	switch (format)
	{
		case BC3_SRGB:
		case BC3_UNORM:
		case BC5_RG:
		case BC6H_UFLOAT:
		case BC6H_SFLOAT:
		case BC7_SRGB:
		case BC7_UNORM: return BC;
		case ASTC_4X4_SRGB:
		case ASTC_4X4_UNORM: return ASTC;
		case EAC_RG: return ETC2;
		default: return NONE;
	}
}

fn VkBlendFactor vk_blend_factor(BlendFactor factor) @private
{
	switch (factor)
	{
		case DEFAULT:   // Resolved by the caller: ONE as a source, ZERO as a destination.
		case ZERO: return VK_BLEND_FACTOR_ZERO;
		case ONE: return VK_BLEND_FACTOR_ONE;
		case SOURCE_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
		case ONE_MINUS_SOURCE_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
		case DESTINATION_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
		case ONE_MINUS_DESTINATION_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
		case SOURCE_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
		case ONE_MINUS_SOURCE_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		case DESTINATION_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
		case ONE_MINUS_DESTINATION_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
		case SOURCE_ALPHA_SATURATE: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
	}
}

// BlendOp, StencilOp, LoadOp, StoreOp, IndexType and AddressMode have Vulkan's values. CompareOp and Filter do not: zero is the useful value.
fn VkCompareOp vk_compare_op(CompareOp op) @private
{
	switch (op)
	{
		case DEFAULT:   // The depth test and comparison samplers; a stencil face resolves it to ALWAYS first.
		case LESS_EQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
		case NEVER: return VK_COMPARE_OP_NEVER;
		case LESS: return VK_COMPARE_OP_LESS;
		case EQUAL: return VK_COMPARE_OP_EQUAL;
		case GREATER: return VK_COMPARE_OP_GREATER;
		case NOT_EQUAL: return VK_COMPARE_OP_NOT_EQUAL;
		case GREATER_EQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
		case ALWAYS: return VK_COMPARE_OP_ALWAYS;
	}
}

fn VkFilter vk_filter(Filter filter) @private => filter == LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

fn VkSamplerMipmapMode vk_mipmap_mode(Filter filter) @private => filter == LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;

fn VkPipelineStageFlags2 vk_stages(Stage stages) @private
{
	VkPipelineStageFlags2 result;
	if (stages.indirect) result |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
	if (stages.index_input) result |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
	if (stages.vertex) result |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
	if (stages.task) result |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
	if (stages.mesh) result |= VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
	if (stages.depth_stencil_tests) result |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
	if (stages.fragment) result |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
	if (stages.color_output) result |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
	if (stages.compute) result |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	if (stages.transfer) result |= VK_PIPELINE_STAGE_2_COPY_BIT;
	if (stages.host) result |= VK_PIPELINE_STAGE_2_HOST_BIT;
	if (stages.all_commands) result |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	return result;
}

fn VkAccessFlags2 vk_accesses(Access accesses) @private
{
	VkAccessFlags2 result;
	if (accesses.transfer_read) result |= VK_ACCESS_2_TRANSFER_READ_BIT;
	if (accesses.transfer_write) result |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
	if (accesses.shader_read) result |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT;
	if (accesses.shader_write) result |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	if (accesses.color_read) result |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
	if (accesses.color_write) result |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
	if (accesses.depth_stencil_read) result |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
	if (accesses.depth_stencil_write) result |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	if (accesses.indirect_read) result |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
	if (accesses.index_read) result |= VK_ACCESS_2_INDEX_READ_BIT;
	if (accesses.host_read) result |= VK_ACCESS_2_HOST_READ_BIT;
	if (accesses.descriptor_read) result |= VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT | VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT;
	return result;
}

fn bool is_usable_memory_type(VkPhysicalDeviceMemoryProperties* properties, uint index) @private
{
	VkMemoryType* type = &properties.memoryTypes[index];
	if (type.propertyFlags & FORBIDDEN_MEMORY) return false;
	return (properties.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM) == 0;
}

// ---- Objects --------------------------------------------------------------------------------------------------------------------

struct BackingBuffer
{
	VkBuffer buffer;
	VkDeviceMemory memory;
	void* mapped;
	VkDeviceAddress address;
}

struct GpuHeapOwner
{
	Device* device;
	BackingBuffer backing;
	BufferRecord* record;   // In the device's address map, when it has one.
}

// The descriptors are written to the CPU array first and copied to the mapped heap, which is only ever written: the CPU does not read
// GPU memory.
struct DescriptorHeap
{
	Device* device;
	GpuHeap storage;
	char* descriptors;
	uint capacity;
	ulong stride;   // The size of a descriptor.
}

struct TextureDescriptorHeap
{
	inline DescriptorHeap heap;
}

struct SamplerDescriptorHeap
{
	inline DescriptorHeap heap;
}

struct TextureHeapOwner
{
	Device* device;
	VkDeviceMemory memory;
}

struct TimelineSemaphore
{
	Device* device;
	VkSemaphore semaphore;
}

struct Texture
{
	Device* device;
	VkImage image;
	uint width;
	uint height;
	uint depth;
	uint layer_count;
	TextureType type;
	Format format;
}

struct RenderView
{
	Device* device;
	VkImageView view;
	uint width;
	uint height;
	bool swapchain_view;
}

struct Pso
{
	Device* device;
	VkPipeline pipeline;
	VkPipelineBindPoint bind_point;
}

struct CommandBuffer
{
	Device* device;
	VkFns* vk;
	CommandBuffer* next;
	uint family_index;     // The queue family of its pool: it is submitted to a queue of that family.
	VkCommandPool command_pool;
	VkCommandBuffer command_buffer;
	VkCommandBuffer epilogue;
	VkQueryPool timestamp_pool;
	ulong** timestamp_destinations;
	ulong* timestamp_results;
	uint timestamp_count;
	Swapchain* swapchain;
	bool ended;
	bool submitted;     // Buffers are one time submit: a second submission is a bug.
	bool has_epilogue;
}

struct CommandPool
{
	Device* device;
	uint family_index;
	VkCommandPool command_pool;
	CommandBuffer* first;
	CommandBuffer* last;
	CommandBuffer* next_buffer;
	bool timestamps;
}

struct Queue
{
	VkQueue queue;
	uint family_index;
	bool timestamps;
	List{VkCommandBufferSubmitInfo} command_infos;   // Scratch for submit_commands, kept between submissions.
	List{VkSemaphoreSubmitInfo} wait_infos;
}

struct PresentContext
{
	VkSemaphore acquired;
	VkSemaphore rendered;
	VkFence presented;
	VkSwapchainKHR swapchain;
	bool present_pending;
}

struct DeferredSwapchainImage
{
	ulong retire_value;
	VkSwapchainKHR swapchain;
	VkImageView view;
}

struct RetiredSwapchain
{
	VkSwapchainKHR handle;
	VkImageView[MAX_SWAPCHAIN_IMAGES] views;
	uint view_count;
}

struct Swapchain
{
	Device* device;
	VkSwapchainKHR handle;
	VkImage[MAX_SWAPCHAIN_IMAGES] images;
	RenderView[MAX_SWAPCHAIN_IMAGES] render_views;
	bool[MAX_SWAPCHAIN_IMAGES] initialized;
	uint image_count;
	uint image_index;
	uint width;
	uint height;
	Format format;
	VkColorSpaceKHR color_space;
	VkSurfaceTransformFlagBitsKHR transform;
	VkCompositeAlphaFlagBitsKHR composite_alpha;
	PresentContext* present_context;
	CommandBuffer* transition_commands;
	bool acquired;
	bool recreate_required;
}

struct Device
{
	os::DylibHandle loader;
	VkFns vk;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debug_messenger;
	VkPhysicalDevice physical_device;
	VkDevice device;
	Queue* queues;
	uint queue_count;
	uint[QUEUE_TYPE_COUNT] queue_type_counts;   // General, compute only, copy only: the queues are in this order.
	VkSurfaceKHR surface;
	uint[QUEUE_TYPE_COUNT] queue_families;
	uint queue_family_count;
	uint timestamp_query_count;
	VkPhysicalDeviceMemoryProperties memory_properties;
	VkPhysicalDeviceProperties physical_properties;
	VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_properties;
	ulong max_timeline_value_difference;
	ulong texture_heap_alignment;
	uint texture_memory_type;
	BufferAddressMap* buffer_map;   // Without VK_KHR_device_address_commands: addresses resolve to buffers and offsets through this.
	DeviceCaps caps;
	char[256] name;
	VkFormatFeatureFlags2[FORMAT_COUNT] format_features;
	bool texture_compression_etc2;
	bool swapchain_color_space_enabled;
	VkSemaphore presentation_retirement;
	ulong presentation_retirement_value;
	ulong completed_presentation_retirement;
	Deque{DeferredSwapchainImage} retired_images;   // Swapchain images, in the order of the presentation timeline values they wait for.
	PresentContext[MAX_SWAPCHAIN_IMAGES] present_contexts;
	RetiredSwapchain[MAX_SWAPCHAIN_IMAGES] retired_swapchains;
	Swapchain* swapchain;
	Swapchain* acquired_swapchain;
	uint present_context_count;
	uint next_present_context;
	bool presentation;
	SurfaceProvider surface_provider;
}

// ---- Device memory --------------------------------------------------------------------------------------------------------------

<*
 The best memory type among `bits` that has `required`, preferring `preferred` and avoiding `avoided`, in a heap of at least
 `minimum_heap_size`.
*>
fn bool Device.find_memory_type(&self, uint bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, VkDeviceSize minimum_heap_size,
	uint* output, VkMemoryPropertyFlags avoided = 0)
{
	bool has_best;
	bool best_is_avoided;
	uint best;
	uint best_score;
	VkDeviceSize best_heap_size;
	for (uint i = 0; i < self.memory_properties.memoryTypeCount; i++)
	{
		if ((bits & (1u << i)) == 0) continue;
		VkMemoryPropertyFlags flags = self.memory_properties.memoryTypes[i].propertyFlags;
		if ((flags & required) != required) continue;
		if (!is_usable_memory_type(&self.memory_properties, i)) continue;
		VkMemoryHeap* heap = &self.memory_properties.memoryHeaps[self.memory_properties.memoryTypes[i].heapIndex];
		if (heap.size < minimum_heap_size) continue;
		bool is_avoided = (flags & avoided) != 0;
		uint score = (uint)(flags & preferred).popcount();
		if (!has_best || (best_is_avoided && !is_avoided) || (best_is_avoided == is_avoided && (score > best_score || (score == best_score && heap.size > best_heap_size))))
		{
			best = i;
			has_best = true;
			best_is_avoided = is_avoided;
			best_score = score;
			best_heap_size = heap.size;
		}
	}
	if (!has_best) return false;
	*output = best;
	return true;
}

fn VkSharingMode Device.sharing_mode(&self) => self.queue_family_count > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;

// A buffer that every queue family can use.
fn VkBufferCreateInfo Device.buffer_info(&self, VkDeviceSize size, VkBufferUsageFlags usage) => {
	.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	.size = size,
	.usage = usage,
	.sharingMode = self.sharing_mode(),
	.queueFamilyIndexCount = self.queue_family_count,
	.pQueueFamilyIndices = &self.queue_families[0],
};

fn BackingBuffer? Device.create_backing_buffer(&self, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
	VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags avoided = 0)
{
	VkFns* vk = &self.vk;
	BackingBuffer result;
	VkBufferCreateInfo buffer_info = self.buffer_info(size, usage);
	check_vk(vk.vkCreateBuffer(self.device, &buffer_info, null, &result.buffer), "vkCreateBuffer", size)!;
	defer catch vk.vkDestroyBuffer(self.device, result.buffer, null);

	VkMemoryRequirements requirements;
	vk.vkGetBufferMemoryRequirements(self.device, result.buffer, &requirements);
	uint memory_type;
	bool host_visible = (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
	if (!self.find_memory_type(requirements.memoryTypeBits, required, preferred, requirements.size, &memory_type, avoided))
	{
		report(string::tformat("No compatible %s GPU memory heap can hold %.2f MiB.", host_visible ? "CPU-visible" : "device-local", (double)requirements.size / (1024.0 * 1024.0)));
		return OUT_OF_MEMORY~;
	}
	VkMemoryAllocateFlagsInfo flags_info = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT };
	VkMemoryAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = &flags_info,
		.allocationSize = requirements.size,
		.memoryTypeIndex = memory_type,
	};
	check_vk(vk.vkAllocateMemory(self.device, &allocate_info, null, &result.memory), host_visible ? "vkAllocateMemory (CPU-visible GPU heap)" : "vkAllocateMemory (GPU-only heap)", requirements.size)!;
	defer catch vk.vkFreeMemory(self.device, result.memory, null);
	check_vk(vk.vkBindBufferMemory(self.device, result.buffer, result.memory, 0), "vkBindBufferMemory", requirements.size)!;
	if (host_visible) check_vk(vk.vkMapMemory(self.device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped), "vkMapMemory", requirements.size)!;
	VkBufferDeviceAddressInfo address_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = result.buffer };
	result.address = vk.vkGetBufferDeviceAddress(self.device, &address_info);
	return result;
}

fn GpuHeap? Device.allocate_gpu_heap(&self, VkDeviceSize size, MemoryType memory)
{
	VkMemoryPropertyFlags required;
	VkMemoryPropertyFlags preferred;
	VkMemoryPropertyFlags avoided;
	switch (memory)
	{
		case CPU_VISIBLE:
			required = CPU_VISIBLE_MEMORY;
		case GPU_ONLY:
			required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			avoided = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		case READBACK:
			required = CPU_VISIBLE_MEMORY;
			preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	}
	BackingBuffer backing = self.create_backing_buffer(size, UNIVERSAL_BUFFER_USAGE, required, preferred, avoided)!;
	GpuHeapOwner* heap = mem::new(GpuHeapOwner, { .device = self, .backing = backing });
	if (self.buffer_map) heap.record = self.buffer_map.insert(backing.address, size, (ulong)backing.buffer);
	return { .cpu = (char*)backing.mapped, .gpu = (char*)(uptr)backing.address, .size = size, .owner = heap };
}

<*
 A descriptor heap lives in CPU visible device memory with the reserved range that the extension wants after the descriptors. The
 returned heap starts at the alignment that the extension requires, so the backing allocation has that much padding.
*>
fn GpuHeap? Device.allocate_descriptor_heap(&self, VkDeviceSize size, DescriptorHeapType type)
{
	VkDeviceSize resource_alignment = max(self.heap_properties.imageDescriptorAlignment, self.heap_properties.bufferDescriptorAlignment);
	bool texture = type == TEXTURE;
	VkDeviceSize reserved_alignment = texture ? resource_alignment : self.heap_properties.samplerDescriptorAlignment;
	VkDeviceSize heap_alignment = texture ? self.heap_properties.resourceHeapAlignment : self.heap_properties.samplerHeapAlignment;
	VkDeviceSize reserved_size = texture ? self.heap_properties.minResourceHeapReservedRange : self.heap_properties.minSamplerHeapReservedRange;

	VkDeviceSize reserved_offset = align_up(size, reserved_alignment);
	VkDeviceSize bind_size = reserved_offset + reserved_size;
	VkDeviceSize allocation_alignment = max(heap_alignment, (VkDeviceSize)GPU_ALLOCATION_ALIGNMENT);
	VkDeviceSize backing_size = bind_size + allocation_alignment - 1;

	BackingBuffer backing = self.create_backing_buffer(backing_size, UNIVERSAL_BUFFER_USAGE | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, CPU_VISIBLE_MEMORY, 0)!;
	GpuHeapOwner* heap = mem::new(GpuHeapOwner, { .device = self, .backing = backing });
	VkDeviceAddress gpu_address = align_up(backing.address, allocation_alignment);
	VkDeviceSize allocation_offset = gpu_address - backing.address;
	return { .cpu = (char*)backing.mapped + allocation_offset, .gpu = (char*)(uptr)gpu_address, .size = size, .owner = heap };
}

// ---- Probing the device ---------------------------------------------------------------------------------------------------------

fn VkMemoryRequirements Device.buffer_memory_requirements(&self, VkBufferUsageFlags usage)
{
	VkBufferCreateInfo buffer_info = self.buffer_info(1, usage);
	VkDeviceBufferMemoryRequirements requirements_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS, .pCreateInfo = &buffer_info };
	VkMemoryRequirements2 requirements = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
	self.vk.vkGetDeviceBufferMemoryRequirements(self.device, &requirements_info, &requirements);
	return requirements.memoryRequirements;
}

fn bool Device.supports_gpu_heap_memory(&self)
{
	VkMemoryRequirements ordinary = self.buffer_memory_requirements(UNIVERSAL_BUFFER_USAGE);
	uint memory_type;
	if (!self.find_memory_type(ordinary.memoryTypeBits, CPU_VISIBLE_MEMORY, 0, ordinary.size, &memory_type)
		|| !self.find_memory_type(ordinary.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, ordinary.size, &memory_type, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
	{
		return false;
	}
	VkMemoryRequirements descriptor = self.buffer_memory_requirements(UNIVERSAL_BUFFER_USAGE | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT);
	return self.find_memory_type(descriptor.memoryTypeBits, CPU_VISIBLE_MEMORY, 0, descriptor.size, &memory_type);
}

fn bool fits_image_format_properties(VkImageCreateInfo* image_info, VkImageFormatProperties* properties) @private
{
	return image_info.extent.width <= properties.maxExtent.width && image_info.extent.height <= properties.maxExtent.height
		&& image_info.extent.depth <= properties.maxExtent.depth && image_info.mipLevels <= properties.maxMipLevels
		&& image_info.arrayLayers <= properties.maxArrayLayers;
}

fn bool Device.supports_image_create_info(&self, VkImageCreateInfo* image_info, VkImageFormatProperties* output = null)
{
	VkPhysicalDeviceImageFormatInfo2 format_info = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
		.format = image_info.format,
		.type = image_info.imageType,
		.tiling = image_info.tiling,
		.usage = image_info.usage,
		.flags = image_info.flags,
	};
	VkImageFormatProperties2 properties = { .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
	VkResult result = self.vk.vkGetPhysicalDeviceImageFormatProperties2(self.physical_device, &format_info, &properties);
	if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) return false;
	if (catch check_vk(result, "vkGetPhysicalDeviceImageFormatProperties2")) return false;
	if (output) *output = properties.imageFormatProperties;
	return fits_image_format_properties(image_info, &properties.imageFormatProperties);
}

fn VkMemoryRequirements Device.image_memory_requirements(&self, VkImageCreateInfo* image_info)
{
	VkDeviceImageMemoryRequirements requirements_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = image_info };
	VkMemoryRequirements2 requirements = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
	self.vk.vkGetDeviceImageMemoryRequirements(self.device, &requirements_info, &requirements);
	return requirements.memoryRequirements;
}

fn void Device.include_texture_heap_alignment(&self, VkMemoryRequirements* requirements)
{
	if (requirements.alignment > self.texture_heap_alignment) self.texture_heap_alignment = requirements.alignment;
}

<*
 Picks the one device local memory type that textures live in, from what the images of the supported profile agree on. Probes DCC capable
 color, broad 3D, and the depth and stencil formats.
*>
fn bool Device.select_texture_memory_type(&self)
{
	VkFormatFeatureFlags2 color_features = self.format_features[Format.RGBA8_UNORM.ordinal];
	if ((color_features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT) == 0) return false;
	uint probe_2d_size = min(self.physical_properties.limits.maxImageDimension2D, 2048u);
	VkImageUsageFlags color_usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	if (color_features & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT) color_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_UNORM,
		.extent = { .width = probe_2d_size, .height = probe_2d_size, .depth = 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = color_usage,
		.sharingMode = self.sharing_mode(),
		.queueFamilyIndexCount = self.queue_family_count,
		.pQueueFamilyIndices = &self.queue_families[0],
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	if (!self.supports_image_create_info(&image_info))
	{
		image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
		if (!self.supports_image_create_info(&image_info)) return false;
	}
	VkMemoryRequirements color_requirements = self.image_memory_requirements(&image_info);
	uint memory_type_bits = color_requirements.memoryTypeBits;
	self.include_texture_heap_alignment(&color_requirements);

	TextureUsage broad_texture_usage = { .sampled = true, .storage = true, .transfer_destination = true };
	VkFormatFeatureFlags2 broad_features = self.format_features[Format.RGBA32_FLOAT.ordinal];
	VkFormatFeatureFlags2 broad_required_features = required_format_features(broad_texture_usage);
	if ((broad_features & broad_required_features) == broad_required_features)
	{
		uint probe_3d_size = min(self.physical_properties.limits.maxImageDimension3D, 2048u);
		image_info.imageType = VK_IMAGE_TYPE_3D;
		image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
		image_info.extent = { .width = probe_3d_size, .height = probe_3d_size, .depth = min(probe_3d_size, 4u) };
		image_info.mipLevels = 32 - (uint)probe_3d_size.clz();
		image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		if (self.supports_image_create_info(&image_info))
		{
			VkMemoryRequirements requirements = self.image_memory_requirements(&image_info);
			self.include_texture_heap_alignment(&requirements);
		}
	}

	Format[5] depth_stencil_formats = { D16_UNORM, D24_UNORM_S8_UINT, D32_FLOAT, S8_UINT, D32_FLOAT_S8_UINT };
	VkFormatFeatureFlags2 storage_features = required_format_features({ .storage = true });
	foreach (format : depth_stencil_formats)
	{
		VkFormatFeatureFlags2 features = self.format_features[format.ordinal];
		TextureFormatInfo info = get_texture_format_info(format);
		bool combined = info.depth && info.stencil;
		VkImageUsageFlags compatibility_usage;
		if (features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT)
		{
			compatibility_usage = VK_IMAGE_USAGE_SAMPLED_BIT;
		}
		else if (features & VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			compatibility_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		}
		else if ((features & storage_features) == storage_features)
		{
			compatibility_usage = VK_IMAGE_USAGE_STORAGE_BIT;
		}
		else if (!combined && (features & VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT))
		{
			compatibility_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		}
		else if (!combined && (features & VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT))
		{
			compatibility_usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}
		if (compatibility_usage == 0) continue;
		image_info.imageType = VK_IMAGE_TYPE_2D;
		image_info.format = vk_format(format);
		image_info.extent = { .width = 1, .height = 1, .depth = 1 };
		image_info.mipLevels = 1;
		image_info.usage = compatibility_usage;
		VkImageFormatProperties compatibility_properties;
		if (!self.supports_image_create_info(&image_info, &compatibility_properties)) continue;
		image_info.extent = { .width = 512, .height = 512, .depth = 1 };
		if ((features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT) && (features & VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT))
		{
			image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		}
		bool supported = image_info.usage == compatibility_usage
			? fits_image_format_properties(&image_info, &compatibility_properties)
			: self.supports_image_create_info(&image_info);
		if (!supported && image_info.usage != compatibility_usage)
		{
			image_info.usage = compatibility_usage;
			supported = fits_image_format_properties(&image_info, &compatibility_properties);
		}
		if (!supported)
		{
			image_info.extent = { .width = 1, .height = 1, .depth = 1 };
			image_info.usage = compatibility_usage;
		}
		VkMemoryRequirements requirements = self.image_memory_requirements(&image_info);
		memory_type_bits &= requirements.memoryTypeBits;
		self.include_texture_heap_alignment(&requirements);
	}
	return self.find_memory_type(memory_type_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, 1, &self.texture_memory_type, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
}

fn VkFormatFeatureFlags2 Device.optimal_format_features(&self, Format format)
{
	VkFormatProperties3 properties3 = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 };
	VkFormatProperties2 properties2 = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &properties3 };
	self.vk.vkGetPhysicalDeviceFormatProperties2(self.physical_device, vk_format(format), &properties2);
	return properties3.optimalTilingFeatures;
}
