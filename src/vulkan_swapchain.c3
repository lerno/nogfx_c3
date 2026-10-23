module nogfx @feat(!DARWIN, NOGFX_VULKAN);
import std::io, std::os, libc;

// ---- Swapchain ------------------------------------------------------------------------------------------------------------------
//
// One swapchain per windowed device, rebuilt when the surface changes size. A retired swapchain stays alive until the presentation
// engine and the GPU are done with it, see the presentation bookkeeping in vulkan_device.c3.

macro void @require_ok(#result, String operation) @private
{
	if (catch err = #result)
	{
		report(string::tformat("%s failed.", operation), true);
		libc::abort();
	}
}

fn void retire_swapchain_handle(Swapchain* swapchain) @private
{
	Device* device = swapchain.device;
	if (!swapchain.handle)
	{
		assert(swapchain.image_count == 0);
		swapchain.width = 0;
		swapchain.height = 0;
		return;
	}
	device.poll_present_contexts();
	RetiredSwapchain retired = { .handle = swapchain.handle };
	for (uint i = 0; i < swapchain.image_count; i++)
	{
		RenderView* render_view = &swapchain.render_views[i];
		retired.views[retired.view_count++] = render_view.view;
		*render_view = {};
		swapchain.images[i] = {};
		swapchain.initialized[i] = false;
	}
	swapchain.image_count = 0;
	swapchain.handle = {};
	swapchain.width = 0;
	swapchain.height = 0;

	if (device.is_presenting(retired.handle))
	{
		foreach (&slot : device.retired_swapchains)
		{
			if (slot.handle) continue;
			*slot = retired;
			return;
		}
		foreach (&context : device.present_contexts[:device.present_context_count])
		{
			if (context.swapchain == retired.handle) device.wait_present_context(context);
		}
	}
	device.queue_retired_swapchain(&retired);
}

fn void destroy_owned_swapchain(Device* device) @private
{
	if (!device.swapchain) return;
	Swapchain* swapchain = device.swapchain;
	assert(!swapchain.acquired && device.acquired_swapchain != swapchain);
	retire_swapchain_handle(swapchain);
	swapchain.device = null;
	mem::free(swapchain);
	device.swapchain = null;
}

fn VkCompositeAlphaFlagBitsKHR choose_composite_alpha(VkCompositeAlphaFlagsKHR supported) @private
{
	VkCompositeAlphaFlagBitsKHR[4] choices = {
		VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
	};
	foreach (choice : choices)
	{
		if (supported & choice) return choice;
	}
	return 0;
}

fn Uint2 resolve_surface_extent(Device* device, VkSurfaceCapabilitiesKHR* caps) @private
{
	if (caps.currentExtent.width != uint::max && caps.currentExtent.height != uint::max)
	{
		return { caps.currentExtent.width, caps.currentExtent.height };
	}
	if (device.surface_provider.get_extent)
	{
		Uint2 size = device.surface_provider.get_extent(device.surface_provider.user_data);
		if (size.x == 0 || size.y == 0) return {};
		return {
			min(max(size.x, caps.minImageExtent.width), caps.maxImageExtent.width),
			min(max(size.y, caps.minImageExtent.height), caps.maxImageExtent.height),
		};
	}
	return { caps.minImageExtent.width, caps.minImageExtent.height };
}

fn bool swapchain_surface_configuration_changed(Swapchain* swapchain) @private
{
	Device* device = swapchain.device;
	VkSurfaceCapabilitiesKHR capabilities;
	@require_ok(check_vk(device.vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical_device, device.surface, &capabilities),
		"vkGetPhysicalDeviceSurfaceCapabilitiesKHR"), "Display surface query");
	Uint2 target = resolve_surface_extent(device, &capabilities);
	bool size_changed = target.x != swapchain.width || target.y != swapchain.height;
	return size_changed || capabilities.currentTransform != swapchain.transform || choose_composite_alpha(capabilities.supportedCompositeAlpha) != swapchain.composite_alpha;
}

<*
 Builds the swapchain for the surface's current extent, replacing the old one. A zero extent leaves it empty until acquire sees a
 nonzero one.
*>
fn void? recreate_swapchain(Swapchain* swapchain) @private
{
	Device* device = swapchain.device;
	VkFns* vk = &device.vk;
	assert(!swapchain.acquired && !device.acquired_swapchain);
	VkPresentModeKHR present_mode = SWAPCHAIN_PRESENT_MODE;
	VkSurfacePresentModeKHR present_mode_info = { .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR, .presentMode = present_mode };
	VkPhysicalDeviceSurfaceInfo2KHR surface_info = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
		.pNext = &present_mode_info,
		.surface = device.surface,
	};
	VkSurfaceCapabilities2KHR capabilities_info = { .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR };
	check_vk(vk.vkGetPhysicalDeviceSurfaceCapabilities2KHR(device.physical_device, &surface_info, &capabilities_info), "vkGetPhysicalDeviceSurfaceCapabilities2KHR")!;
	VkSurfaceCapabilitiesKHR* capabilities = &capabilities_info.surfaceCapabilities;

	Uint2 resolved = resolve_surface_extent(device, capabilities);
	VkExtent2D extent = { resolved.x, resolved.y };
	if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0)
	{
		report("Surface does not support color attachment usage.");
		return UNSUPPORTED~;
	}

	uint surface_format_count;
	check_vk(vk.vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical_device, device.surface, &surface_format_count, null), "vkGetPhysicalDeviceSurfaceFormatsKHR")!;
	if (surface_format_count == 0)
	{
		report("Surface returned zero supported formats.");
		return UNSUPPORTED~;
	}
	VkSurfaceFormatKHR[] formats = mem::new_array(VkSurfaceFormatKHR, surface_format_count);
	defer mem::free(formats.ptr);
	check_vk(vk.vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical_device, device.surface, &surface_format_count, formats.ptr), "vkGetPhysicalDeviceSurfaceFormatsKHR")!;

	VkFormat requested_format = vk_format(swapchain.format);
	bool format_supported;
	for (uint i = 0; i < surface_format_count; i++)
	{
		if ((formats[i].format == requested_format || formats[i].format == VK_FORMAT_UNDEFINED) && formats[i].colorSpace == swapchain.color_space)
		{
			format_supported = true;
			break;
		}
	}
	if (!format_supported)
	{
		report(string::tformat("Requested swapchain format %d with colorspace %d is unsupported by surface.", (int)requested_format, (int)swapchain.color_space));
		return UNSUPPORTED~;
	}

	if (extent.width == 0 || extent.height == 0)
	{
		swapchain.width = 0;
		swapchain.height = 0;
		swapchain.recreate_required = true;
		return;
	}

	uint requested_image_count = max(device.present_context_count, capabilities.minImageCount);
	if (capabilities.maxImageCount != 0) requested_image_count = min(requested_image_count, capabilities.maxImageCount);
	if (requested_image_count == 0 || requested_image_count > MAX_SWAPCHAIN_IMAGES) return UNSUPPORTED~;

	VkCompositeAlphaFlagBitsKHR composite_alpha = choose_composite_alpha(capabilities.supportedCompositeAlpha);
	if (!composite_alpha)
	{
		report("The display surface exposes no supported composite alpha mode.");
		return UNSUPPORTED~;
	}
	VkSwapchainKHR old_handle = swapchain.handle;
	VkSwapchainPresentModesCreateInfoKHR present_modes_info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR,
		.presentModeCount = 1,
		.pPresentModes = &present_mode,
	};
	VkSwapchainCreateInfoKHR create_info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = &present_modes_info,
		.surface = device.surface,
		.minImageCount = requested_image_count,
		.imageFormat = requested_format,
		.imageColorSpace = swapchain.color_space,
		.imageExtent = extent,
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = device.sharing_mode(),
		.queueFamilyIndexCount = device.queue_family_count,
		.pQueueFamilyIndices = &device.queue_families[0],
		.preTransform = capabilities.currentTransform,
		.compositeAlpha = composite_alpha,
		.presentMode = present_mode,
		.clipped = VK_TRUE,
		.oldSwapchain = old_handle,
	};
	VkSwapchainKHR new_handle;
	// vkCreateSwapchainKHR retires oldSwapchain even when it fails, so any failure from here on retires it.
	defer catch retire_swapchain_handle(swapchain);
	check_vk(vk.vkCreateSwapchainKHR(device.device, &create_info, null, &new_handle), "vkCreateSwapchainKHR")!;
	defer catch vk.vkDestroySwapchainKHR(device.device, new_handle, null);

	VkImage[MAX_SWAPCHAIN_IMAGES] images;
	uint image_count;
	check_vk(vk.vkGetSwapchainImagesKHR(device.device, new_handle, &image_count, null), "vkGetSwapchainImagesKHR")!;
	if (image_count == 0 || image_count > MAX_SWAPCHAIN_IMAGES) return UNSUPPORTED~;
	check_vk(vk.vkGetSwapchainImagesKHR(device.device, new_handle, &image_count, &images[0]), "vkGetSwapchainImagesKHR")!;

	VkImageView[MAX_SWAPCHAIN_IMAGES] views;
	uint view_count;
	defer catch for (uint i = 0; i < view_count; i++) vk.vkDestroyImageView(device.device, views[i], null);
	for (; view_count < image_count; view_count++)
	{
		VkImageViewCreateInfo view_info = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = images[view_count],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = requested_format,
			.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
		};
		check_vk(vk.vkCreateImageView(device.device, &view_info, null, &views[view_count]), "vkCreateImageView (swapchain)")!;
	}

	retire_swapchain_handle(swapchain);
	swapchain.handle = new_handle;
	swapchain.image_count = image_count;
	swapchain.width = extent.width;
	swapchain.height = extent.height;
	swapchain.transform = capabilities.currentTransform;
	swapchain.composite_alpha = composite_alpha;
	swapchain.recreate_required = false;
	for (uint i = 0; i < image_count; i++)
	{
		swapchain.images[i] = images[i];
		swapchain.render_views[i] = { .device = device, .view = views[i], .width = extent.width, .height = extent.height, .swapchain_view = true };
	}
}

<*
 Call after `wait_idle`, outside an acquired frame. An unsupported pair leaves the previous mode usable; a zero size drawable defers
 the recreation until its extent is nonzero. UNSUPPORTED without a window, or for an extended color space that the instance lacks.
*>
fn void? Device.set_swapchain_format(&self, Format format, ColorSpace color_space)
{
	assert(!self.acquired_swapchain);
	if (!self.swapchain || (color_space != SRGB && !self.swapchain_color_space_enabled)) return UNSUPPORTED~;
	Swapchain* swapchain = self.swapchain;
	VkColorSpaceKHR requested_color_space = color_space == EXTENDED_SRGB_LINEAR ? VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	if (format == UNDEFINED) format = BGRA8_UNORM;
	if (swapchain.format == format && swapchain.color_space == requested_color_space) return;

	Format old_format = swapchain.format;
	VkColorSpaceKHR old_color_space = swapchain.color_space;
	swapchain.format = format;
	swapchain.color_space = requested_color_space;
	if (catch err = recreate_swapchain(swapchain))
	{
		swapchain.format = old_format;
		swapchain.color_space = old_color_space;
		// The failed creation has retired the old swapchain: build it again.
		if (!swapchain.handle) @require_ok(recreate_swapchain(swapchain), "Swapchain recreation");
		return err~;
	}
}

<*
 The size of the surface in pixels, zero when there is no window or the surface has no fixed size.
*>
fn Uint2 Device.drawable_extent(&self)
{
	assert(!self.acquired_swapchain, "drawable_extent must be called outside presentation");
	if (!self.swapchain) return {};
	VkSurfaceCapabilitiesKHR capabilities;
	if (catch check_vk(self.vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.physical_device, self.surface, &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")) return {};
	Uint2 extent = resolve_surface_extent(self, &capabilities);
	Swapchain* swapchain = self.swapchain;
	if (swapchain.handle && (swapchain.width != extent.x || swapchain.height != extent.y)) swapchain.recreate_required = true;
	return extent;
}

<*
 Acquires the next swapchain image outside a render pass. The result is empty while the surface extent is zero. A non empty acquire
 must be submitted with `submit_and_present`, on queue zero, with this buffer among the commands. The image is in the GENERAL layout.
*>
fn SwapchainFrame CommandBuffer.acquire(&self)
{
	Device* device = self.device;
	VkFns* vk = self.vk;
	assert(!self.swapchain);
	assert(device.swapchain && !device.swapchain.acquired && !device.acquired_swapchain);
	device.poll_presentation_retirement();

	Swapchain* swapchain = device.swapchain;
	if (swapchain.handle && swapchain_surface_configuration_changed(swapchain))
	{
		swapchain.recreate_required = true;
	}
	PresentContext* present_context;
	for (;;)
	{
		if (!swapchain.handle || swapchain.recreate_required)
		{
			@require_ok(recreate_swapchain(swapchain), "Swapchain recreation");
			bool drawable = swapchain.handle && swapchain.width != 0 && swapchain.height != 0;
			if (!drawable) return {};
		}
		if (!present_context)
		{
			present_context = &device.present_contexts[device.next_present_context];
			device.wait_present_context(present_context);
			assert(!present_context.present_pending && !present_context.swapchain);
		}
		uint image_index;
		VkResult result = vk.vkAcquireNextImageKHR(device.device, swapchain.handle, ulong::max, present_context.acquired, 0, &image_index);
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			swapchain.recreate_required = true;
			continue;
		}
		if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) abort_vk(result, "vkAcquireNextImageKHR");
		assert(image_index < swapchain.image_count, "swapchain returned an invalid image index");

		swapchain.image_index = image_index;
		swapchain.present_context = present_context;
		swapchain.acquired = true;
		swapchain.recreate_required = result == VK_SUBOPTIMAL_KHR && swapchain_surface_configuration_changed(swapchain);
		device.acquired_swapchain = swapchain;
		record_layout_transition(vk, self.command_buffer, swapchain.images[image_index], VK_IMAGE_ASPECT_COLOR_BIT,
			swapchain.initialized[image_index] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
		self.swapchain = swapchain;
		swapchain.transition_commands = self;
		return { .render_view = &swapchain.render_views[image_index], .extent = { swapchain.width, swapchain.height } };
	}
}

<*
 Submits the commands on queue zero and presents the acquired image after them. The submission must include the buffer that acquired it.
*>
fn void Device.submit_and_present(&self, SubmitDesc desc)
{
	assert(desc.commands.len == 0 || desc.commands.ptr);
	Swapchain* swapchain = self.swapchain;
	assert(swapchain && swapchain.acquired && swapchain.transition_commands && swapchain.present_context);
	$if env::COMPILER_SAFE_MODE:
		bool contains_swapchain;
		foreach (commands : desc.commands)
		{
			assert(commands && (!commands.swapchain || commands == swapchain.transition_commands));
			if (commands == swapchain.transition_commands) contains_swapchain = true;
		}
		assert(contains_swapchain, "presentation submission must include the acquired command buffer");
	$endif
	PresentContext* present_context = swapchain.present_context;
	assert(!present_context.present_pending && !present_context.swapchain);
	assert_vk(self.vk.vkResetFences(self.device, 1, &present_context.presented));
	self.submit_commands(desc, 0, present_context.acquired, present_context.rendered);
	swapchain.transition_commands.swapchain = null;
	swapchain.transition_commands = null;
	swapchain.initialized[swapchain.image_index] = true;

	VkSwapchainPresentFenceInfoKHR fence_info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
		.swapchainCount = 1,
		.pFences = &present_context.presented,
	};
	VkPresentInfoKHR present_info = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext = &fence_info,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &present_context.rendered,
		.swapchainCount = 1,
		.pSwapchains = &swapchain.handle,
		.pImageIndices = &swapchain.image_index,
	};
	VkResult result = self.vk.vkQueuePresentKHR(self.queues[0].queue, &present_info);
	present_context.present_pending = true;
	present_context.swapchain = swapchain.handle;
	swapchain.present_context = null;
	swapchain.acquired = false;
	self.acquired_swapchain = null;
	self.next_present_context = (self.next_present_context + 1) % self.present_context_count;
	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		swapchain.recreate_required = true;
	}
	else if (result == VK_SUBOPTIMAL_KHR)
	{
		swapchain.recreate_required = swapchain.recreate_required || swapchain_surface_configuration_changed(swapchain);
	}
	else if (result != VK_SUCCESS)
	{
		abort_vk(result, "vkQueuePresentKHR");
	}
}
