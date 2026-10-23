module triangle_generic;
import std, nogfx;

fn int main()
{
	init_gfx()!!;
	defer gfx::finalize();

	GfxWindow* win = gfx::create_window("Generic Triangle", 200, 200, 900, 600, {})!!;
	defer win.destroy();
	win.set_exit_key(ESCAPE);
	win.set_repaint_callback(&draw);

	Device*? created = create_gpu_device(win);
	if (catch err = created)
	{
		io::printfn("GPU device unavailable (%s), falling back to software renderer.", err);
		use_software = true;
	}
	else
	{
		device = created;
		io::printfn("Using GPU: %s", device.caps.device_name);

		upload = device.create_gpu_heap(64 * 1024, CPU_VISIBLE)!!;
		triangle_pso = create_pipeline(device)!!;

		foreach (&p : pools) *p = device.create_command_pool()!!;
		timeline = device.create_timeline_semaphore()!!;
	}

	defer if (!use_software)
	{
		device.wait_idle();
		timeline.destroy();
		foreach (p : pools) p.destroy();
		triangle_pso.destroy();
		upload.destroy();
		device.destroy();
	};

	while (!win.should_close())
	{
		while (try e = gfx::poll(win))
		{
			if (e.type == KEY_PRESSED && (e.key.value == F11 || e.key.value == F))
			{
				win.set_fullscreen(!win.is_fullscreen());
			}
		}
		draw(win, {});
	}
	return 0;
}

fn void draw(GfxWindow* win, GfxRect rect)
{
	if (use_software)
	{
		draw_software(win);
	}
	else
	{
		draw_gpu(win);
	}
}

struct Vertex
{
	float[<4>] position;
	float[<4>] color;
}

struct DrawRoot
{
	Vertex* vertices;
}

struct FrameData
{
	DrawRoot root @align(16);
	Vertex[3] vertices @align(16);
}

const uint FRAMES = 2;

const Vec2[3] MODEL_POS = { { 0.0f, -0.6f }, { 0.6f, 0.6f }, { -0.6f, 0.6f } };
const Vec4[3] MODEL_COL = { { 1.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } };

bool use_software;
Device* device;
GpuHeap upload;
CommandPool*[FRAMES] pools;
TimelineSemaphore* timeline;
ulong frame_count;
Pso* triangle_pso;

fn void* gpu_address(void* cpu) => upload.gpu + ((char*)cpu - upload.cpu);

fn void draw_gpu(GfxWindow* win)
{
	ulong frame = frame_count + 1;
	if (frame > FRAMES)
	{
		TimelinePoint previous = { timeline, frame - FRAMES };
		previous.wait();
	}
	CommandPool* pool = pools[frame % FRAMES];
	pool.reset();
	CommandBuffer* commands = pool.begin_commands()!!;
	SwapchainFrame target = commands.acquire();
	if (!target.render_view || target.extent.x == 0 || target.extent.y == 0) return;

	float aspect = (float)target.extent.x / (float)target.extent.y;
	Mat4 proj = { .m = { [0] = 1.0f / aspect, [5] = 1.0f, [10] = 1.0f, [15] = 1.0f } };

	FrameData* data = &((FrameData*)upload.cpu)[frame % FRAMES];
	for (sz i = 0; i < 3; i++)
	{
		data.vertices[i] = { proj * (Vec4){ ...MODEL_POS[i], 0.0f, 1.0f }, MODEL_COL[i] };
	}
	data.root.vertices = (Vertex*)gpu_address(&data.vertices);

	commands.begin_render_pass({
		.colors = { { .render_view = target.render_view, .load = CLEAR, .clear = { 0.08f, 0.08f, 0.12f, 1.0f } } }
	});
	commands.bind_pso(triangle_pso);
	commands.draw(gpu_address(&data.root), 3);
	commands.end_render_pass();
	commands.end();

	frame_count = frame;
	device.submit_and_present({ .commands = { commands }, .completion = { timeline, frame } });
}

fn void draw_software(GfxWindow* win) => @pool()
{
	GfxPoint size = win.get_size_in_pixels();
	if (size.x <= 0 || size.y <= 0) size = win.get_size();
	if (size.x <= 0 || size.y <= 0) return;
	int w = size.x;
	int h = size.y;
	float aspect = (float)w / (float)h;

	uint[] pixels = mem::temp_array(uint, (sz)w * h);
	pixels[..] = 0xFF14141E;

	Vec2[3] pts;
	for (sz i = 0; i < 3; i++)
	{
		Vec2 ndc = { MODEL_POS[i].x / aspect, MODEL_POS[i].y };
		pts[i] = (ndc * 0.5f + 0.5f) * (Vec2){ (float)w, (float)h };
	}

	Bounds bb = { .min = math::min(pts[0], math::min(pts[1], pts[2])), .max = math::max(pts[0], math::max(pts[1], pts[2])) };
	bb.min = math::clamp(bb.min, (Vec2){ 0, 0 }, (Vec2){ (float)(w - 1), (float)(h - 1) });
	bb.max = math::clamp(bb.max, (Vec2){ 0, 0 }, (Vec2){ (float)(w - 1), (float)(h - 1) });

	float area = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
	if (area != 0.0f)
	{
		for (int y = (int)bb.min.y; y <= (int)bb.max.y; y++)
		{
			for (int x = (int)bb.min.x; x <= (int)bb.max.x; x++)
			{
				Vec2 p = { (float)x, (float)y };
				Vec3 w_vec = { (pts[1] - p).cross(pts[2] - p), (pts[2] - p).cross(pts[0] - p), (pts[0] - p).cross(pts[1] - p) };
				if (w_vec.comp_ge((Vec3)0).and() || w_vec.comp_le((Vec3)0).and())
				{
					Vec3 col = math::abs(w_vec / area) * 255.0f;
					pixels[y * w + x] = 0xFF000000 | ((uint)col.y << 16) | ((uint)col.z << 8) | (uint)col.x;
				}
			}
		}
	}
	GfxSurface* surf = gfx::create_surface((char*)pixels.ptr, w, h, gfx::native_format(), win)!!;
	defer surf.destroy();
	win.blit_surface(surf);
}


// ## Metal! Backend 🤘️
module triangle_generic @feat(DARWIN & !NOGFX_VULKAN);
import std::gfx, nogfx;

const String SHADER_SOURCE = `
#include <metal_stdlib>
using namespace metal;
struct Vertex { float4 position; float4 color; };
struct DrawRoot { device const Vertex* vertices; };
struct VOut { float4 position [[position]]; float4 color; };
vertex VOut vs(device const DrawRoot* root [[buffer(0)]], uint vid [[vertex_id]]) {
	return { root->vertices[vid].position, root->vertices[vid].color };
}
fragment float4 fs(VOut in [[stage_in]]) { return in.color; }
`;

fn void init_gfx() => gfx::init("Generic Triangle", {})!;

fn Device*? create_gpu_device(GfxWindow* win)
{
	return nogfx::create_device({
		.window = win.create_metal_layer()!,
		.swapchain_format = BGRA8_UNORM,
		.desired_swapchain_image_count = 3,
	});
}

fn Pso*? create_pipeline(Device* dev)
{
	return dev.create_graphics_pso({
		.vertex = { .code = (char[])SHADER_SOURCE, .entry_point = "vs" },
		.fragment = { .code = (char[])SHADER_SOURCE, .entry_point = "fs" },
		.color_targets = { { .format = BGRA8_UNORM } },
	});
}


// Vulkan Backend 🖖️
module triangle_generic @feat(!DARWIN, NOGFX_VULKAN);
import std::gfx, nogfx;

const char[*] TRIANGLE_VERTEX = $embed("../tests/vulkan/triangle.vert.spv");
const char[*] TRIANGLE_FRAGMENT = $embed("../tests/vulkan/triangle.frag.spv");

fn void? init_gfx() => gfx::init("Generic Triangle", { .init_vulkan = true })!;

fn Device*? create_gpu_device(GfxWindow* win)
{
	$if env::WIN32:
		return nogfx::create_device({
			.window = win.src.window,
			.swapchain_format = BGRA8_UNORM,
			.desired_swapchain_image_count = 3,
		});
	$else
		String[] exts = gfx::get_required_vulkan_extensions();
		ZString[] z_exts = mem::temp_array(ZString, exts.len);
		foreach (i, ext : exts) z_exts[i] = ext.zstr_tcopy();

		return nogfx::create_device({
			.surface_provider = {
				.required_extensions = z_exts,
				.user_data = win,
				.create_surface = fn ulong?(void* instance, void* data) {
					return (ulong)((GfxWindow*)data).create_vulkan_surface((std::gfx::VkInstance)instance);
				},
				.get_extent = fn Uint2(void* data) {
					GfxWindow* w = (GfxWindow*)data;
					GfxPoint p = w.get_size_in_pixels();
					if (p.x <= 0 || p.y <= 0) p = w.get_size();
					return { (uint)max(p.x, 0), (uint)max(p.y, 0) };
				},
			},
			.swapchain_format = BGRA8_UNORM,
			.desired_swapchain_image_count = 3,
		});
	$endif
}

fn Pso*? create_pipeline(Device* dev)
{
	return dev.create_graphics_pso({
		.vertex = { .code = TRIANGLE_VERTEX[..] },
		.fragment = { .code = TRIANGLE_FRAGMENT[..] },
		.color_targets = { { .format = BGRA8_UNORM } },
	});
}