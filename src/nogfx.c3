module nogfx;

// A port of NoGraphicsAPI (https://github.com/sebbbi/NoGraphicsAPI, MIT, Sebastian Aaltonen): the public types. The backends are metal.c3
// (Metal 4, on Apple platforms) and vulkan*.c3 (Vulkan 1.4, everywhere else, and on Apple platforms with -D NOGFX_VULKAN). Only one is compiled.
//
// The model: memory is raw GPU pointers into heaps the application owns, a draw or dispatch takes one 16-byte aligned root pointer that
// the shader reads everything else from, textures and samplers are indices into descriptor heaps, and synchronization is barriers by
// stage plus timeline semaphores. There is no hidden state tracking and no validation: resource destruction is immediate and the
// application makes sure that the GPU is done with a resource first.
//
// Differences from the C++ API, all of them because a C3 struct has no field defaults: every struct is zero initialized, and zero is made to
// mean what upstream's default is wherever that can be said:
//  - TextureType.TWO_D, Filter.LINEAR, AddressMode.REPEAT, BlendOp.ADD, StencilOp.KEEP, LoadOp.LOAD and StoreOp.STORE are zero,
//  - BlendFactor.DEFAULT is ONE for a source factor and ZERO for a destination factor, so `{ .enabled = true }` passes the source color
//    through; CompareOp.DEFAULT is LESS_EQUAL for the depth test and a comparison sampler, ALWAYS for a stencil face,
//  - an extent / mip_levels / layer_count of zero is 1, a texture's usage of zero is `sampled` and its format UNDEFINED is RGBA8_UNORM,
//    a ColorTargetDesc.write_mask of zero is all channels, stencil read and write masks of zero are all bits, a ShaderStage.entry_point of
//    null is "main" and a threadgroup_size component of zero is 1,
//  - DeviceDesc: zero queues is one queue, zero swapchain images is two, a swapchain without a format is BGRA8_UNORM, and timestamps are off
//    unless timestamp_query_count is set.
// What cannot be zero and has to be set: the viewport's max_depth (1.0) and size, the scissor's size, a depth attachment's clear value
// (zero unless set; upstream's is 1.0) and a ColorAttachment's clear alpha (zero; upstream's is 1.0).
//
// Where the backends differ the comments say so: Vulkan wants SPIR-V, has no use for threadgroup_size, and presents to an HWND;
// Metal takes a metallib or MSL source and presents to a CAMetalLayer.

faultdef UNSUPPORTED, DEVICE_LOST, DRIVER_ERROR, OUT_OF_MEMORY, CREATION_FAILED;

<*
 Synchronous diagnostics on the calling thread. The message is only valid during the callback. Every error is printed to stderr as well.
 The thread-local callback defaults to null.
*>
alias ErrorCallback = fn void(String message, bool fatal);
tlocal ErrorCallback error_callback;
fn void set_error_callback(ErrorCallback callback) => error_callback = callback;

struct Uint2 { uint x, y; }
struct Uint3 { uint x, y, z; }

enum MemoryType : char { CPU_VISIBLE, GPU_ONLY, READBACK }

struct SizeAlign { ulong size, align; }

struct GpuRange
{
	void* gpu;
	ulong size;
}

<*
 A block of raw memory for application side suballocation. `cpu` is null for GPU_ONLY. Copies alias the same allocation:
 call `destroy` on one unchanged copy exactly once.
*>
struct GpuHeap
{
	char* cpu;
	char* gpu;
	ulong size;
	GpuHeapOwner* owner;
}

<*
 Texture heaps use one GPU only memory type and must outlive every placed texture. Copies alias the same allocation:
 call `destroy` on one unchanged copy exactly once.
*>
struct TextureHeap
{
	ulong size;
	TextureHeapOwner* owner;
}

struct TimelinePoint
{
	TimelineSemaphore* semaphore;
	ulong value;
}

struct SubmitDesc
{
	CommandBuffer*[] commands;
	TimelinePoint[] waits;      // GPU waits cover all command stages.
	TimelinePoint completion;   // Required for every submission.
}

// Zero is UNDEFINED: "inherit the texture format" in a texture descriptor, and BGRA8_UNORM for a swapchain.
enum Format : char
{
	UNDEFINED,

	R8_SRGB,
	RG8_SRGB,
	RGBA8_SRGB,
	BGRA8_SRGB,

	RGBA4_UNORM,
	R5G5B5A1_UNORM,
	R5G6B5_UNORM,

	R8_UNORM,
	RG8_UNORM,
	RGBA8_UNORM,
	BGRA8_UNORM,
	R16_UNORM,
	RG16_UNORM,
	RGBA16_UNORM,

	R8_UINT,
	RG8_UINT,
	RGBA8_UINT,
	BGRA8_UINT,
	R16_UINT,
	RG16_UINT,
	RGBA16_UINT,
	R32_UINT,
	RG32_UINT,
	RGBA32_UINT,

	R16_FLOAT,
	RG16_FLOAT,
	RGBA16_FLOAT,
	R32_FLOAT,
	RG32_FLOAT,
	RGBA32_FLOAT,

	RGB10A2_UNORM,
	RG11B10_FLOAT,

	D16_UNORM,
	D24_UNORM_S8_UINT,
	D32_FLOAT,
	S8_UINT,
	D32_FLOAT_S8_UINT,

	EAC_RG,
	ASTC_4X4_SRGB,
	ASTC_4X4_UNORM,
	BC3_SRGB,
	BC3_UNORM,
	BC5_RG,
	BC6H_UFLOAT,
	BC6H_SFLOAT,
	BC7_SRGB,
	BC7_UNORM,
}

struct TextureFormatInfo
{
	Uint2 block_extent;
	uint bytes_per_block;
	bool depth;
	bool stencil;
}

fn TextureFormatInfo get_texture_format_info(Format format)
{
	switch (format)
	{
		case R8_SRGB:
		case R8_UNORM:
		case R8_UINT:
		case S8_UINT:
			return { .block_extent = { 1, 1 }, .bytes_per_block = 1, .stencil = format == S8_UINT };
		case RG8_SRGB:
		case RGBA4_UNORM:
		case R5G5B5A1_UNORM:
		case R5G6B5_UNORM:
		case RG8_UNORM:
		case R16_UNORM:
		case RG8_UINT:
		case R16_UINT:
		case R16_FLOAT:
		case D16_UNORM:
			return { .block_extent = { 1, 1 }, .bytes_per_block = 2, .depth = format == D16_UNORM };
		case RGBA8_SRGB:
		case BGRA8_SRGB:
		case RGBA8_UNORM:
		case BGRA8_UNORM:
		case RG16_UNORM:
		case RGBA8_UINT:
		case BGRA8_UINT:
		case RG16_UINT:
		case R32_UINT:
		case RG16_FLOAT:
		case R32_FLOAT:
		case RGB10A2_UNORM:
		case RG11B10_FLOAT:
		case D24_UNORM_S8_UINT:
		case D32_FLOAT:
			return { .block_extent = { 1, 1 }, .bytes_per_block = 4, .depth = format == D24_UNORM_S8_UINT || format == D32_FLOAT, .stencil = format == D24_UNORM_S8_UINT };
		case RGBA16_UNORM:
		case RGBA16_UINT:
		case RG32_UINT:
		case RGBA16_FLOAT:
		case RG32_FLOAT:
		case D32_FLOAT_S8_UINT:
			return { .block_extent = { 1, 1 }, .bytes_per_block = 8, .depth = format == D32_FLOAT_S8_UINT, .stencil = format == D32_FLOAT_S8_UINT };
		case RGBA32_UINT:
		case RGBA32_FLOAT:
			return { .block_extent = { 1, 1 }, .bytes_per_block = 16 };
		case EAC_RG:
		case ASTC_4X4_SRGB:
		case ASTC_4X4_UNORM:
		case BC3_SRGB:
		case BC3_UNORM:
		case BC5_RG:
		case BC6H_UFLOAT:
		case BC6H_SFLOAT:
		case BC7_SRGB:
		case BC7_UNORM:
			return { .block_extent = { 4, 4 }, .bytes_per_block = 16 };
		case UNDEFINED:
			return {};
	}
}

enum TextureType : char { TWO_D, ONE_D, THREE_D, CUBE, TWO_D_ARRAY, CUBE_ARRAY }

bitstruct TextureUsage : uint
{
	bool sampled : 0;
	bool storage : 1;
	bool color_attachment : 2;
	bool depth_stencil_attachment : 3;
	bool transfer_source : 4;
	bool transfer_destination : 5;
}

enum TextureDescriptorType : char { SAMPLED, STORAGE }
enum TextureAspect : char { AUTOMATIC, COLOR, DEPTH, STENCIL }
enum Filter : char { LINEAR, NEAREST }
enum AddressMode : char { REPEAT, MIRRORED_REPEAT, CLAMP_TO_EDGE }
// DEFAULT, the zero value, is LESS_EQUAL for the depth test and for a comparison sampler and ALWAYS for a stencil face, as upstream's defaults.
enum CompareOp : char { DEFAULT, LESS_EQUAL, NEVER, LESS, EQUAL, GREATER, NOT_EQUAL, GREATER_EQUAL, ALWAYS }
fn CompareOp CompareOp.or(self, CompareOp fallback) => self == DEFAULT ? fallback : self;
enum CullMode : char { NONE, CLOCKWISE, COUNTER_CLOCKWISE }
// DEFAULT, the zero value, is ONE as a source factor and ZERO as a destination factor.
enum BlendFactor : char
{
	DEFAULT,
	ZERO,
	ONE,
	SOURCE_COLOR,
	ONE_MINUS_SOURCE_COLOR,
	DESTINATION_COLOR,
	ONE_MINUS_DESTINATION_COLOR,
	SOURCE_ALPHA,
	ONE_MINUS_SOURCE_ALPHA,
	DESTINATION_ALPHA,
	ONE_MINUS_DESTINATION_ALPHA,
	SOURCE_ALPHA_SATURATE,
}
fn BlendFactor BlendFactor.or(self, BlendFactor fallback) => self == DEFAULT ? fallback : self;
enum BlendOp : char { ADD, SUBTRACT, REVERSE_SUBTRACT, MINIMUM, MAXIMUM }
enum IndexType : char { UINT16, UINT32 }
enum LoadOp : char { LOAD, CLEAR, DISCARD }
enum StoreOp : char { STORE, DISCARD }
enum StencilOp : char { KEEP, ZERO, REPLACE, INCREMENT_CLAMP, DECREMENT_CLAMP, INVERT, INCREMENT_WRAP, DECREMENT_WRAP }

<*
 Ordered by Vulkan's logical execution order where stages are comparable. A barrier's before scope includes the selected and logically
 earlier stages; its after scope includes the selected and logically later stages. Vertex and task / mesh are alternative graphics
 branches, depth_stencil_tests spans early and late tests around fragment, compute and transfer are separate pipelines, host is a
 pseudo stage (a barrier destination only, paired with Access.host_read), and all_commands is all GPU command stages but not host.
*>
bitstruct Stage : ulong
{
	bool indirect : 0;
	bool index_input : 1;
	bool vertex : 2;
	bool task : 3;
	bool mesh : 4;
	bool depth_stencil_tests : 5;
	bool fragment : 6;
	bool color_output : 7;
	bool compute : 8;
	bool transfer : 9;
	bool host : 10;
	bool all_commands : 11;
}

bitstruct Access : ulong
{
	bool transfer_read : 0;
	bool transfer_write : 1;
	bool shader_read : 2;
	bool shader_write : 3;
	bool color_read : 4;
	bool color_write : 5;
	bool depth_stencil_read : 6;
	bool depth_stencil_write : 7;
	bool indirect_read : 8;
	bool index_read : 9;
	bool host_read : 10;
	bool descriptor_read : 11;
}

struct DeviceCaps
{
	String device_name;
	// Flat queue indices: general first, then compute only, then copy only. Queue zero supports presentation.
	uint queue_count;
	uint general_queue_count;
	uint compute_queue_count;
	uint copy_queue_count;
	// Copy only texture offsets and extents align to these texel block counts, except at mip edges. Zero means whole mip levels only.
	Uint3 copy_texture_granularity;
	ulong max_push_data_size;
	// Common element size for suballocating TextureHeap storage; every SizeAlign.align divides this value.
	ulong texture_heap_alignment;
	float timestamp_period_ns;   // Nanoseconds per timestamp tick; zero when profiling is unavailable.
	uint sub_texel_precision_bits;
	bool texture_compression_bc;
	bool texture_compression_astc;
	bool storage_input_output16;
	bool indirect_mesh_draw;
}

enum ColorSpace : uint { SRGB, EXTENDED_SRGB_LINEAR }

alias SurfaceCreatorFn = fn ulong?(void* instance, void* user_data);
alias SurfaceExtentFn = fn Uint2(void* user_data);

struct SurfaceProvider
{
	ZString[] required_extensions;
	SurfaceCreatorFn create_surface;
	SurfaceExtentFn get_extent;
	void* user_data;
}

struct DeviceDesc
{
	void* window;                       // A CAMetalLayer* (Metal) or an HWND (Vulkan Win32). Null creates a headless device.
	SurfaceProvider surface_provider;   // Vulkan surface provider for platforms without a native HWND (Linux/BSD).
	Format swapchain_format;            // Zero is BGRA8_UNORM. Metal: BGRA8_UNORM or BGRA8_SRGB. Vulkan: any format the surface offers.
	ColorSpace swapchain_color_space;
	uint desired_swapchain_image_count; // Zero is 2. Metal clamps to 2..3 drawables; Vulkan takes 1..8 presentation contexts.
	bool steam_overlay_compatibility;   // Metal on macOS only: allow drawable texture views and present from a classic Metal queue, which Steam's overlay hooks.
	uint desired_queue_count;           // General graphics + compute queues. Zero is 1.
	uint desired_compute_queue_count;
	uint desired_copy_queue_count;
	uint timestamp_query_count;         // Per command buffer. Zero disables timestamps.
}

struct SwapchainFrame
{
	RenderView* render_view;
	Uint2 extent;
}

struct TextureDesc
{
	TextureType type;
	Uint3 extent;
	uint mip_levels;
	uint layer_count;       // Array layers; cube faces are individual layers.
	Format format;          // UNDEFINED is RGBA8_UNORM.
	bool mutable_format;    // Allow format compatible descriptor views, but could lose DCC.
	TextureUsage usage;     // Zero is `sampled`.
}

struct RenderViewDesc
{
	uint mip_level;
	uint slice;     // Physical array slice; cube faces are individual slices.
}

struct TextureDescriptorDesc
{
	Format format;          // UNDEFINED inherits the texture format.
	TextureAspect aspect;   // AUTOMATIC selects color, or depth before stencil.
	uint base_mip;
	uint mip_count;         // Zero selects every remaining mip level.
	uint base_layer;
	uint layer_count;       // Zero selects every remaining layer.
}

struct TextureCopyDesc
{
	uint mip_level;
	uint base_slice;
	uint slice_count;           // Zero selects every remaining physical slice.
	Uint3 offset;
	Uint3 extent;               // Zero components select the remaining mip extent.
	ulong row_pitch_bytes;      // Zero is tightly packed.
	ulong slice_pitch_bytes;    // Zero is tightly packed.
}

struct SamplerDesc
{
	Filter min_filter;
	Filter mag_filter;
	Filter mip_filter;
	AddressMode address_u;
	AddressMode address_v;
	AddressMode address_w;
	bool anisotropic;       // Uses the API's fixed 4x profile.
	bool compare_enabled;
	CompareOp compare;
}

struct BlendComponentState
{
	BlendFactor source;
	BlendFactor destination;
	BlendOp operation;
}

struct BlendState
{
	bool enabled;
	BlendComponentState color;
	BlendComponentState alpha;
}

struct ColorTargetDesc
{
	Format format;
	BlendState blend;
	char write_mask;    // Bit 0 red .. bit 3 alpha. Zero is all channels.
}

struct RasterizationState
{
	CullMode cull;
	float depth_bias_constant;
	float depth_bias_clamp;
	float depth_bias_slope;
}

struct Viewport
{
	float x, y, width, height;
	float min_depth, max_depth;
}

struct Scissor
{
	int x, y;
	uint width, height;
}

struct StencilFaceState
{
	CompareOp compare;
	StencilOp fail;
	StencilOp pass;
	StencilOp depth_fail;
	char reference;
}

struct DepthStencilState
{
	bool depth_test;
	bool depth_write;
	CompareOp depth_compare;
	bool stencil_test;
	char stencil_read_mask;    // Zero is all bits. A mask of no bits is never useful: keep the value with the StencilOps instead.
	char stencil_write_mask;   // Zero is all bits.
	StencilFaceState front;
	StencilFaceState back;
}

struct ShaderStage
{
	char[] code;                // Metal: a metallib (starting with MTLB), or Metal Shading Language source compiled when the pipeline is created.
	                            // Vulkan: SPIR-V (any alignment: it is copied if it is not 4 byte aligned). The root is the uniform buffer at set 0, binding 0.
	ZString entry_point;        // Null is "main".
	Uint3 threadgroup_size;     // Metal compute, task and mesh: must match the compiled numthreads, a zero component is 1. Vulkan reads it from the SPIR-V.
}

struct GraphicsPSODesc
{
	ShaderStage vertex;
	ShaderStage fragment;   // Empty omits the fragment stage, for depth only rasterization.
	ColorTargetDesc[] color_targets;
	Format depth_format;
	Format stencil_format;
	RasterizationState rasterization;
}

struct MeshPSODesc
{
	ShaderStage task;       // Empty launches mesh workgroups directly; otherwise draws launch task workgroups.
	ShaderStage mesh;
	ShaderStage fragment;
	ColorTargetDesc[] color_targets;
	Format depth_format;
	Format stencil_format;
	RasterizationState rasterization;
}

struct ClearColor { float x, y, z, w; }

struct ColorAttachment
{
	RenderView* render_view;
	LoadOp load;
	StoreOp store;
	ClearColor clear;
}

struct DepthAttachment
{
	RenderView* render_view;
	LoadOp load;
	StoreOp store;
	float clear;
}

struct StencilAttachment
{
	RenderView* render_view;
	LoadOp load;
	StoreOp store;
	char clear;
}

struct RenderingDesc
{
	ColorAttachment[] colors;
	DepthAttachment depth;
	StencilAttachment stencil;
}

bitstruct RenderingFlags : uint
{
	bool suspending : 1;
	bool resuming : 2;
}
