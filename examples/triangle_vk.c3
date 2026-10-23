module triangle_vk;
import std::gfx, std::io, std::time, nogfx;

const char[*] TRIANGLE_VERTEX = $embed("../tests/vulkan/triangle.vert.spv");
const char[*] TRIANGLE_FRAGMENT = $embed("../tests/vulkan/triangle.frag.spv");

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

Device* device;
GpuHeap upload;
CommandPool*[FRAMES] pools;
TimelineSemaphore* timeline;
ulong frame_count;
Pso* triangle_pso;

fn void* gpu_address(void* cpu) => upload.gpu + ((char*)cpu - upload.cpu);

fn void draw(GfxWindow* win, GfxRect rect)
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

	FrameData* data = &((FrameData*)upload.cpu)[frame % FRAMES];
	data.vertices[0] = { {  0.0f / aspect, -0.6f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } };
	data.vertices[1] = { {  0.6f / aspect,  0.6f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } };
	data.vertices[2] = { { -0.6f / aspect,  0.6f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } };

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

fn int main()
{
	gfx::init("NoGraphicsAPI Vulkan Triangle", { .init_vulkan = true })!!;
	defer gfx::finalize();

	GfxWindow* win = gfx::create_window("NoGraphicsAPI Vulkan Triangle", 200, 200, 900, 600, {})!!;
	defer win.destroy();

	String[] exts = gfx::get_required_vulkan_extensions();
	ZString[] z_exts = mem::temp_array(ZString, exts.len);
	foreach (i, ext : exts) z_exts[i] = ext.zstr_tcopy();

	SurfaceProvider provider = {
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
	};

	device = nogfx::create_device({
		.surface_provider = provider,
		.swapchain_format = BGRA8_UNORM,
		.desired_swapchain_image_count = 3,
	})!!;
	defer device.destroy();
	io::printfn("Using %s", device.caps.device_name);

	upload = device.create_gpu_heap(64 * 1024, CPU_VISIBLE)!!;
	defer upload.destroy();

	triangle_pso = device.create_graphics_pso({
		.vertex = { .code = TRIANGLE_VERTEX[..] },
		.fragment = { .code = TRIANGLE_FRAGMENT[..] },
		.color_targets = { { .format = BGRA8_UNORM } },
	})!!;
	defer triangle_pso.destroy();

	foreach (&p : pools) *p = device.create_command_pool()!!;
	defer foreach (p : pools) p.destroy();

	timeline = device.create_timeline_semaphore()!!;
	defer timeline.destroy();

	win.set_repaint_callback(&draw);

	bool quit = false;
	while (!quit && !win.should_close())
	{
		while (try e = gfx::poll(win))
		{
			switch (e.type)
			{
				case WINDOW_CLOSE:
					quit = true;
				case KEY_PRESSED:
					if (e.key.value == ESCAPE) quit = true;
					if (e.key.value == F11 || e.key.value == F) win.set_fullscreen(!win.is_fullscreen());
				default:
					break;
			}
		}
		draw(win, {});
	}
	device.wait_idle();
	return 0;
}