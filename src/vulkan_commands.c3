module nogfx @feat(!DARWIN, NOGFX_VULKAN);
import std::io, std::os, std::collections, libc;

// ---- Pipelines ------------------------------------------------------------------------------------------------------------------

struct Spirv
{
	uint* words;
	usz size;
	void* copy;   // Set when the code was not 4 byte aligned and had to be copied.
}

<*
 SPIR-V must be 4 byte aligned for Vulkan, which a C3 byte slice (for instance an $embed) need not be: copy it when it is not.
*>
fn Spirv load_spirv(char[] code) @private
{
	assert(code.len >= 20 && code.len % 4 == 0, "PSO stages require SPIR-V binaries");
	Spirv result = { .size = code.len };
	if ((uptr)code.ptr % 4 != 0)
	{
		result.copy = mem::malloc(code.len);
		mem::copy(result.copy, code.ptr, code.len);
		result.words = (uint*)result.copy;
	}
	else
	{
		result.words = (uint*)code.ptr;
	}
	assert(*result.words == 0x07230203, "PSO stage does not contain SPIR-V");
	return result;
}

// Binding zero of set zero is the root: a uniform buffer (or read only storage buffer) whose address is the pointer in the push data.
fn VkDescriptorSetAndBindingMappingEXT root_mapping() @private => {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
	.bindingCount = 1,
	.resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT | VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT,
	.source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT,
	.sourceData = { .pushAddressOffset = 0 },
};

// One shader stage with everything its create info chains to: the SPIR-V inline (no shader module) and the root mapping. The pointers point
// into the struct, so it must stay where `init` was called.
struct StageInfo
{
	Spirv code;
	VkDescriptorSetAndBindingMappingEXT root;
	VkShaderModuleCreateInfo module_info;
	VkShaderDescriptorSetAndBindingMappingInfoEXT mapping;
	VkPipelineShaderStageCreateInfo stage;
}

fn void StageInfo.init(&self, ShaderStage source, VkShaderStageFlagBits kind)
{
	self.code = load_spirv(source.code);
	self.root = root_mapping();
	self.module_info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = self.code.size, .pCode = self.code.words };
	self.mapping = {
		.sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
		.pNext = &self.module_info,
		.mappingCount = 1,
		.pMappings = &self.root,
	};
	self.stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &self.mapping, .stage = kind, .pName = source.entry_point ?: "main" };
}

fn Pso*? create_raster_pso(Device* device, ShaderStage first_stage, ShaderStage fragment, ColorTargetDesc[] color_targets,
	Format depth_format, Format stencil_format, RasterizationState rasterization_state, bool mesh, ShaderStage task = {}) @private
{
	assert(color_targets.len <= MAX_COLOR_ATTACHMENTS, "color targets must fit the attachment array");
	StageInfo[3] stages;   // [task] first [fragment]
	uint stage_count;
	defer foreach (&stage : stages[:stage_count]) mem::free(stage.code.copy);
	if (task.code.len) stages[stage_count++].init(task, VK_SHADER_STAGE_TASK_BIT_EXT);
	stages[stage_count++].init(first_stage, mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT);
	if (fragment.code.len) stages[stage_count++].init(fragment, VK_SHADER_STAGE_FRAGMENT_BIT);
	VkPipelineShaderStageCreateInfo[3] stage_infos;
	foreach (index, &stage : stages[:stage_count]) stage_infos[index] = stage.stage;

	VkPipelineVertexInputStateCreateInfo vertex_input = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
	VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	VkPipelineViewportStateCreateInfo viewport_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
	bool depth_bias = rasterization_state.depth_bias_constant != 0.0f || rasterization_state.depth_bias_clamp != 0.0f || rasterization_state.depth_bias_slope != 0.0f;
	VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = rasterization_state.cull == NONE ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT,
		.frontFace = rasterization_state.cull == COUNTER_CLOCKWISE ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = depth_bias ? VK_TRUE : VK_FALSE,
		.depthBiasConstantFactor = rasterization_state.depth_bias_constant,
		.depthBiasClamp = rasterization_state.depth_bias_clamp,
		.depthBiasSlopeFactor = rasterization_state.depth_bias_slope,
		.lineWidth = 1.0f,
	};
	VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	VkPipelineDepthStencilStateCreateInfo depth_stencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	VkPipelineColorBlendAttachmentState[MAX_COLOR_ATTACHMENTS] color_attachments;
	VkFormat[MAX_COLOR_ATTACHMENTS] color_formats;
	foreach (index, &target : color_targets)
	{
		color_attachments[index] = {
			.blendEnable = target.blend.enabled ? VK_TRUE : VK_FALSE,
			.srcColorBlendFactor = vk_blend_factor(target.blend.color.source.or(ONE)),
			.dstColorBlendFactor = vk_blend_factor(target.blend.color.destination.or(ZERO)),
			.colorBlendOp = (VkBlendOp)target.blend.color.operation.ordinal,
			.srcAlphaBlendFactor = vk_blend_factor(target.blend.alpha.source.or(ONE)),
			.dstAlphaBlendFactor = vk_blend_factor(target.blend.alpha.destination.or(ZERO)),
			.alphaBlendOp = (VkBlendOp)target.blend.alpha.operation.ordinal,
			.colorWriteMask = (VkColorComponentFlags)(target.write_mask ?: 0xF),
		};
		color_formats[index] = vk_format(target.format);
	}
	VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = (uint)color_targets.len,
		.pAttachments = color_targets.len ? &color_attachments[0] : null,
	};
	VkDynamicState[10] dynamic_states = {
		VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
		VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
		VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
		VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
		VK_DYNAMIC_STATE_STENCIL_OP,
		VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
		VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
		VK_DYNAMIC_STATE_STENCIL_REFERENCE,
	};
	VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = (uint)dynamic_states.len,
		.pDynamicStates = &dynamic_states[0],
	};
	VkPipelineRenderingCreateInfo rendering_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = (uint)color_targets.len,
		.pColorAttachmentFormats = color_targets.len ? &color_formats[0] : null,
		.depthAttachmentFormat = vk_format(depth_format),
		.stencilAttachmentFormat = vk_format(stencil_format),
	};
	VkPipelineCreateFlags2CreateInfo flags_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
		.pNext = &rendering_info,
		.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
	};
	VkGraphicsPipelineCreateInfo pso_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &flags_info,
		.stageCount = stage_count,
		.pStages = &stage_infos[0],
		.pVertexInputState = mesh ? null : &vertex_input,
		.pInputAssemblyState = mesh ? null : &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.basePipelineIndex = -1,
	};
	Pso* result = mem::new(Pso, { .device = device, .bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS });
	defer catch mem::free(result);
	check_vk(device.vk.vkCreateGraphicsPipelines(device.device, 0, 1, &pso_info, null, &result.pipeline), "vkCreateGraphicsPipelines")!;
	return result;
}

<*
 A vertex and fragment PSO, from SPIR-V. The shaders read their data through the root pointer; there is no vertex input. The viewport,
 scissor and depth / stencil state are set on the command buffer.
*>
fn Pso*? Device.create_graphics_pso(&self, GraphicsPSODesc desc)
{
	return create_raster_pso(self, desc.vertex, desc.fragment, desc.color_targets, desc.depth_format, desc.stencil_format, desc.rasterization, false);
}

<*
 A mesh PSO, with an optional task stage in front of the mesh stage.
*>
fn Pso*? Device.create_mesh_pso(&self, MeshPSODesc desc)
{
	return create_raster_pso(self, desc.mesh, desc.fragment, desc.color_targets, desc.depth_format, desc.stencil_format, desc.rasterization, true, desc.task);
}

fn Pso*? Device.create_compute_pso(&self, ShaderStage stage)
{
	StageInfo shader;
	shader.init(stage, VK_SHADER_STAGE_COMPUTE_BIT);
	defer mem::free(shader.code.copy);
	VkPipelineCreateFlags2CreateInfo flags_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
		.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
	};
	VkComputePipelineCreateInfo pso_info = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext = &flags_info,
		.stage = shader.stage,
		.basePipelineIndex = -1,
	};
	Pso* result = mem::new(Pso, { .device = self, .bind_point = VK_PIPELINE_BIND_POINT_COMPUTE });
	defer catch mem::free(result);
	check_vk(self.vk.vkCreateComputePipelines(self.device, 0, 1, &pso_info, null, &result.pipeline), "vkCreateComputePipelines")!;
	return result;
}

fn void Pso.destroy(&self)
{
	self.device.vk.vkDestroyPipeline(self.device.device, self.pipeline, null);
	mem::free(self);
}

// ---- Command pools and buffers --------------------------------------------------------------------------------------------------

<*
 A pool of command buffers for one queue. Buffers recorded from it can be submitted to the queues of the same family. Not thread safe: one
 pool per recording thread and in-flight frame.
*>
fn CommandPool*? Device.create_command_pool(&self, uint queue_index = 0)
{
	assert(queue_index < self.queue_count, "create_command_pool requires an available queue index");
	CommandPool* pool = mem::new(CommandPool, { .device = self, .family_index = self.queues[queue_index].family_index, .timestamps = self.queues[queue_index].timestamps });
	VkCommandPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = self.queues[queue_index].family_index,
	};
	defer catch mem::free(pool);
	check_vk(self.vk.vkCreateCommandPool(self.device, &pool_info, null, &pool.command_pool), "vkCreateCommandPool")!;
	return pool;
}

fn void CommandPool.destroy(&self)
{
	VkFns* vk = &self.device.vk;
	while (self.first)
	{
		CommandBuffer* commands = self.first;
		assert(!commands.swapchain, "an acquired swapchain image must be presented before destroying its command pool");
		self.first = commands.next;
		if (commands.timestamp_pool) vk.vkDestroyQueryPool(self.device.device, commands.timestamp_pool, null);
		mem::free(commands.timestamp_destinations);
		mem::free(commands.timestamp_results);
		mem::free(commands);
	}
	vk.vkDestroyCommandPool(self.device.device, self.command_pool, null);
	mem::free(self);
}

<*
 Recycles the buffers: call only after every buffer submitted from this pool has completed. Unsubmitted buffers are discarded and all
 earlier CommandBuffer handles are invalid.
*>
fn void CommandPool.reset(&self)
{
	for (CommandBuffer* commands = self.first; commands; commands = commands.next)
	{
		assert(!commands.swapchain, "an acquired swapchain image must be presented before resetting its command pool");
		commands.submitted = false;
		commands.ended = false;
		commands.timestamp_count = 0;
	}
	assert_vk(self.device.vk.vkResetCommandPool(self.device.device, self.command_pool, 0));
	self.next_buffer = self.first;
}

<*
 Writes the timestamps of the submitted buffers to the destinations given to `write_timestamp`. Call after every submission from this
 pool has completed and before resetting it. Does not wait.
*>
fn void CommandPool.read_timestamps(&self)
{
	for (CommandBuffer* commands = self.first; commands; commands = commands.next)
	{
		if (!commands.submitted || commands.timestamp_count == 0) continue;
		assert_vk(self.device.vk.vkGetQueryPoolResults(self.device.device, commands.timestamp_pool, 0, commands.timestamp_count,
			ulong::size * commands.timestamp_count, commands.timestamp_results, ulong::size, VK_QUERY_RESULT_64_BIT));
		for (uint i = 0; i < commands.timestamp_count; i++) *commands.timestamp_destinations[i] = commands.timestamp_results[i];
		commands.timestamp_count = 0;
	}
}

fn VkResult Device.allocate_command_buffer(&self, VkCommandPool pool, VkCommandBuffer* output)
{
	VkCommandBufferAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	return self.vk.vkAllocateCommandBuffers(self.device, &allocate_info, output);
}

fn void Device.begin_one_time_commands(&self, VkCommandBuffer command_buffer)
{
	VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	assert_vk(self.vk.vkBeginCommandBuffer(command_buffer, &begin_info));
}

// A command buffer that is not one of the pool's yet, with the query pool for its timestamps if it has them.
fn CommandBuffer*? CommandPool.allocate(&self)
{
	Device* device = self.device;
	CommandBuffer* commands = mem::new(CommandBuffer, { .device = device, .vk = &device.vk, .family_index = self.family_index, .command_pool = self.command_pool });
	defer catch mem::free(commands);
	check_vk(device.allocate_command_buffer(self.command_pool, &commands.command_buffer), "vkAllocateCommandBuffers")!;
	if (self.timestamps)
	{
		defer catch device.vk.vkFreeCommandBuffers(device.device, self.command_pool, 1, &commands.command_buffer);
		VkQueryPoolCreateInfo query_info = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = device.timestamp_query_count };
		check_vk(device.vk.vkCreateQueryPool(device.device, &query_info, null, &commands.timestamp_pool), "vkCreateQueryPool")!;
		commands.timestamp_destinations = mem::new_array(ulong*, device.timestamp_query_count);
		commands.timestamp_results = mem::new_array(ulong, device.timestamp_query_count);
	}
	CommandBuffer** link = self.last ? &self.last.next : &self.first;
	*link = commands;
	self.last = commands;
	return commands;
}

fn CommandBuffer*? CommandPool.begin_commands(&self)
{
	Device* device = self.device;
	CommandBuffer* commands = self.next_buffer;
	if (commands)
	{
		self.next_buffer = commands.next;
	}
	else
	{
		commands = self.allocate()!;
	}
	assert(!commands.swapchain);
	device.begin_one_time_commands(commands.command_buffer);
	commands.timestamp_count = 0;
	commands.submitted = false;
	commands.ended = false;
	commands.has_epilogue = false;
	if (commands.timestamp_pool) device.vk.vkResetQueryPool(device.device, commands.timestamp_pool, 0, device.timestamp_query_count);
	return commands;
}

<*
 Ends recording. A buffer that acquired a swapchain image gets a second, hidden buffer that moves the image to the presentation layout;
 the submission runs it after all the commands.
*>
fn void CommandBuffer.end(&self)
{
	VkCommandBuffer command_buffer = self.command_buffer;
	VkFns* vk = self.vk;
	if (self.swapchain)
	{
		assert_vk(vk.vkEndCommandBuffer(command_buffer));
		if (!self.epilogue) require_vk(self.device.allocate_command_buffer(self.command_pool, &self.epilogue), "vkAllocateCommandBuffers (epilogue)");
		command_buffer = self.epilogue;
		self.device.begin_one_time_commands(command_buffer);
		self.has_epilogue = true;
		Swapchain* swapchain = self.swapchain;
		assert(swapchain.acquired && swapchain.transition_commands == self);
		record_layout_transition(vk, command_buffer, swapchain.images[swapchain.image_index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
	}
	assert_vk(vk.vkEndCommandBuffer(command_buffer));
	self.ended = true;
}

// ---- Submission -----------------------------------------------------------------------------------------------------------------

fn VkCommandBufferSubmitInfo command_submit_info(VkCommandBuffer command_buffer) @private => {
	.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
	.commandBuffer = command_buffer,
	.deviceMask = 1,
};

fn VkSemaphoreSubmitInfo semaphore_submit_info(VkSemaphore semaphore, ulong value = 0) @private => {
	.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
	.semaphore = semaphore,
	.value = value,
	.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
};

<*
 Submits to the queue: waits for the timeline points, runs the commands, signals the completion point. With a presentation (the semaphores
 are the present context's) it also waits for the acquire and signals both the render semaphore and the presentation retirement timeline.
*>
fn void Device.submit_commands(&self, SubmitDesc desc, uint queue_index, VkSemaphore wait_semaphore, VkSemaphore signal_semaphore)
{
	Queue* queue = &self.queues[queue_index];
	TimelineSemaphore* completion = desc.completion.semaphore;
	assert(completion, "every submission needs a completion point");
	queue.command_infos.clear();
	queue.wait_infos.clear();
	foreach (commands : desc.commands)
	{
		assert(commands);
		assert(commands.device == self, "a command buffer of another device");
		assert(commands.family_index == queue.family_index, "a command buffer submitted to a queue of another family than its pool's");
		assert(commands.ended && !commands.submitted, "command buffers are ended, and submitted once: they are one time submit");
		queue.command_infos.push(command_submit_info(commands.command_buffer));
	}
	// Presentation transitions must follow every suspended / resumed segment.
	foreach (commands : desc.commands)
	{
		if (commands.has_epilogue) queue.command_infos.push(command_submit_info(commands.epilogue));
	}
	foreach (point : desc.waits)
	{
		assert(point.semaphore);
		queue.wait_infos.push(semaphore_submit_info(point.semaphore.semaphore, point.value));
	}
	if (wait_semaphore) queue.wait_infos.push(semaphore_submit_info(wait_semaphore));
	VkSemaphoreSubmitInfo[3] signal_infos;
	signal_infos[0] = semaphore_submit_info(completion.semaphore, desc.completion.value);
	uint signal_count = 1;
	if (signal_semaphore)
	{
		signal_infos[signal_count++] = semaphore_submit_info(signal_semaphore);
		signal_infos[signal_count++] = semaphore_submit_info(self.presentation_retirement, self.next_presentation_retirement());
	}
	VkSubmitInfo2 submit_info = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.waitSemaphoreInfoCount = (uint)queue.wait_infos.size,
		.pWaitSemaphoreInfos = queue.wait_infos.entries,
		.commandBufferInfoCount = (uint)queue.command_infos.size,
		.pCommandBufferInfos = queue.command_infos.entries,
		.signalSemaphoreInfoCount = signal_count,
		.pSignalSemaphoreInfos = &signal_infos[0],
	};
	assert_vk(self.vk.vkQueueSubmit2(queue.queue, 1, &submit_info, 0));
	foreach (commands : desc.commands) commands.submitted = true;
}

<*
 Submits ended command buffers to a queue, each exactly once before its pool is reset. Order the completion values across queues.
 Buffers that acquired a swapchain image go through `submit_and_present` instead.
*>
fn void Device.submit(&self, SubmitDesc desc, uint queue_index = 0)
{
	assert(queue_index < self.queue_count, "submit requires an available queue index");
	assert(desc.commands.len == 0 || desc.commands.ptr);
	$if env::COMPILER_SAFE_MODE:
		foreach (commands : desc.commands) assert(commands && !commands.swapchain, "swapchain commands require submit_and_present");
	$endif
	self.submit_commands(desc, queue_index, 0, 0);
}

// ---- Commands -------------------------------------------------------------------------------------------------------------------

fn VkBindHeapInfoEXT heap_bind_info(GpuRange heap, VkDeviceSize reserved_alignment, VkDeviceSize reserved_size) @private
{
	VkDeviceSize reserved_offset = align_up(heap.size, reserved_alignment);
	return {
		.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
		.heapRange = { .address = (VkDeviceAddress)(uptr)heap.gpu, .size = reserved_offset + reserved_size },
		.reservedRangeOffset = reserved_offset,
		.reservedRangeSize = reserved_size,
	};
}

fn void CommandBuffer.set_texture_descriptor_heap(&self, TextureDescriptorHeap* heap)
{
	assert(heap && self.device == heap.device);
	VkPhysicalDeviceDescriptorHeapPropertiesEXT* properties = &self.device.heap_properties;
	VkBindHeapInfoEXT bind_info = heap_bind_info(heap.storage.range(), max(properties.imageDescriptorAlignment, properties.bufferDescriptorAlignment), properties.minResourceHeapReservedRange);
	self.vk.vkCmdBindResourceHeapEXT(self.command_buffer, &bind_info);
}

fn void CommandBuffer.set_sampler_descriptor_heap(&self, SamplerDescriptorHeap* heap)
{
	assert(heap && self.device == heap.device);
	VkPhysicalDeviceDescriptorHeapPropertiesEXT* properties = &self.device.heap_properties;
	VkBindHeapInfoEXT bind_info = heap_bind_info(heap.storage.range(), properties.samplerDescriptorAlignment, properties.minSamplerHeapReservedRange);
	self.vk.vkCmdBindSamplerHeapEXT(self.command_buffer, &bind_info);
}

// Without VK_KHR_device_address_commands: the buffer that holds the range, and the offset of the range in it.
fn VkBuffer CommandBuffer.resolve(&self, GpuRange range, ulong* offset) => (VkBuffer)self.device.buffer_map.resolve(range, offset);

fn VkDeviceAddressRangeKHR address_range(GpuRange range) @private => { .address = (VkDeviceAddress)(uptr)range.gpu, .size = range.size };

fn void CommandBuffer.copy_memory(&self, GpuRange source, GpuRange destination)
{
	if (self.device.buffer_map)
	{
		ulong source_offset;
		ulong destination_offset;
		VkBuffer source_buffer = self.resolve(source, &source_offset);
		VkBuffer destination_buffer = self.resolve(destination, &destination_offset);
		VkBufferCopy2 buffer_region = { .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .srcOffset = source_offset, .dstOffset = destination_offset, .size = source.size };
		VkCopyBufferInfo2 buffer_info = {
			.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
			.srcBuffer = source_buffer,
			.dstBuffer = destination_buffer,
			.regionCount = 1,
			.pRegions = &buffer_region,
		};
		self.vk.vkCmdCopyBuffer2(self.command_buffer, &buffer_info);
		return;
	}
	VkDeviceMemoryCopyKHR region = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_COPY_KHR,
		.srcRange = address_range(source),
		.srcFlags = ADDRESS_FLAGS,
		.dstRange = address_range(destination),
		.dstFlags = ADDRESS_FLAGS,
	};
	VkCopyDeviceMemoryInfoKHR info = { .sType = VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_INFO_KHR, .regionCount = 1, .pRegions = &region };
	self.vk.vkCmdCopyMemoryKHR(self.command_buffer, &info);
}

fn VkDeviceMemoryImageCopyKHR texture_copy_region(Texture* texture, TextureCopyDesc copy, GpuRange memory) @private
{
	uint mip_width = max(texture.width >> copy.mip_level, 1u);
	uint mip_height = max(texture.height >> copy.mip_level, 1u);
	uint mip_depth = max(texture.depth >> copy.mip_level, 1u);
	uint width = copy.extent.x == 0 ? mip_width - copy.offset.x : copy.extent.x;
	TextureFormatInfo format_info = get_texture_format_info(texture.format);
	ulong row_pitch = copy.row_pitch_bytes == 0 ? divide_up(width, format_info.block_extent.x) * format_info.bytes_per_block : copy.row_pitch_bytes;
	assert(copy.slice_pitch_bytes == 0 || row_pitch != 0, "slice pitch conversion requires a non-zero row pitch");
	return {
		.sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
		.addressRange = address_range(memory),
		.addressFlags = ADDRESS_FLAGS,
		.addressRowLength = (uint)(copy.row_pitch_bytes / format_info.bytes_per_block * format_info.block_extent.x),
		.addressImageHeight = copy.slice_pitch_bytes == 0 ? 0 : (uint)(copy.slice_pitch_bytes / row_pitch * format_info.block_extent.y),
		.imageSubresource = {
			.aspectMask = image_aspects(texture.format),
			.mipLevel = copy.mip_level,
			.baseArrayLayer = copy.base_slice,
			.layerCount = copy.slice_count == 0 ? texture.layer_count - copy.base_slice : copy.slice_count,
		},
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.imageOffset = { (int)copy.offset.x, (int)copy.offset.y, (int)copy.offset.z },
		.imageExtent = {
			.width = width,
			.height = copy.extent.y == 0 ? mip_height - copy.offset.y : copy.extent.y,
			.depth = copy.extent.z == 0 ? mip_depth - copy.offset.z : copy.extent.z,
		},
	};
}

fn VkBufferImageCopy2 buffer_texture_copy_region(VkDeviceMemoryImageCopyKHR* region, ulong offset) @private => {
	.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
	.bufferOffset = offset,
	.bufferRowLength = region.addressRowLength,
	.bufferImageHeight = region.addressImageHeight,
	.imageSubresource = region.imageSubresource,
	.imageOffset = region.imageOffset,
	.imageExtent = region.imageExtent,
};

<*
 Copies from memory to a texture. Depth / stencil copies need a general queue; a copy queue also needs copy_texture_granularity alignment.
*>
fn void CommandBuffer.copy_memory_to_texture(&self, GpuRange source, Texture* destination, TextureCopyDesc copy = {})
{
	VkDeviceMemoryImageCopyKHR region = texture_copy_region(destination, copy, source);
	if (self.device.buffer_map || !self.vk.vkCmdCopyMemoryToImageKHR)
	{
		ulong offset;
		VkBuffer buffer = self.resolve(source, &offset);
		VkBufferImageCopy2 buffer_region = buffer_texture_copy_region(&region, offset);
		VkCopyBufferToImageInfo2 buffer_info = {
			.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
			.srcBuffer = buffer,
			.dstImage = destination.image,
			.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
			.regionCount = 1,
			.pRegions = &buffer_region,
		};
		self.vk.vkCmdCopyBufferToImage2(self.command_buffer, &buffer_info);
		return;
	}
	ulong offset;
	VkBuffer buffer = self.resolve(source, &offset);
	VkBufferImageCopy2 buffer_region = buffer_texture_copy_region(&region, offset);
	VkCopyBufferToImageInfo2 buffer_info = {
		.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
		.srcBuffer = buffer,
		.dstImage = destination.image,
		.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.regionCount = 1,
		.pRegions = &buffer_region,
	};
	self.vk.vkCmdCopyBufferToImage2(self.command_buffer, &buffer_info);
}

fn void CommandBuffer.copy_texture_to_memory(&self, Texture* source, GpuRange destination, TextureCopyDesc copy = {})
{
	VkDeviceMemoryImageCopyKHR region = texture_copy_region(source, copy, destination);
	if (self.device.buffer_map || !self.vk.vkCmdCopyImageToMemoryKHR)
	{
		ulong offset;
		VkBuffer buffer = self.resolve(destination, &offset);
		VkBufferImageCopy2 buffer_region = buffer_texture_copy_region(&region, offset);
		VkCopyImageToBufferInfo2 buffer_info = {
			.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
			.srcImage = source.image,
			.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
			.dstBuffer = buffer,
			.regionCount = 1,
			.pRegions = &buffer_region,
		};
		self.vk.vkCmdCopyImageToBuffer2(self.command_buffer, &buffer_info);
		return;
	}
	ulong offset;
	VkBuffer buffer = self.resolve(destination, &offset);
	VkBufferImageCopy2 buffer_region = buffer_texture_copy_region(&region, offset);
	VkCopyImageToBufferInfo2 buffer_info = {
		.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
		.srcImage = source.image,
		.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.dstBuffer = buffer,
		.regionCount = 1,
		.pRegions = &buffer_region,
	};
	self.vk.vkCmdCopyImageToBuffer2(self.command_buffer, &buffer_info);
}

<*
 One global memory barrier from `before` to `after`. It names no resource: describe the hazard, and the stages and accesses that it needs.
*>
fn void CommandBuffer.barrier(&self, Stage before, Access before_access, Stage after, Access after_access)
{
	VkMemoryBarrier2 memory_barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.srcStageMask = vk_stages(before),
		.srcAccessMask = vk_accesses(before_access),
		.dstStageMask = vk_stages(after),
		.dstAccessMask = vk_accesses(after_access),
	};
	VkDependencyInfo dependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memory_barrier };
	self.vk.vkCmdPipelineBarrier2(self.command_buffer, &dependency);
}

<*
 Captures a timestamp, written to `cpu_destination` by `CommandPool.read_timestamps`. Ignored when timestamps are off or the queue has no
 64 bit counters. `stage` must be a single pipeline stage.
*>
fn void CommandBuffer.write_timestamp(&self, ulong* cpu_destination, Stage stage = { .all_commands = true })
{
	if (!self.timestamp_pool) return;
	assert(self.timestamp_count < self.device.timestamp_query_count);
	assert(cpu_destination);
	self.timestamp_destinations[self.timestamp_count] = cpu_destination;
	self.vk.vkCmdWriteTimestamp2(self.command_buffer, vk_stages(stage), self.timestamp_pool, self.timestamp_count++);
}

// ---- Rendering ------------------------------------------------------------------------------------------------------------------

fn void CommandBuffer.set_viewport(&self, Viewport viewport)
{
	VkViewport vk_viewport = {
		.x = viewport.x,
		.y = viewport.y,
		.width = viewport.width,
		.height = viewport.height,
		.minDepth = viewport.min_depth,
		.maxDepth = viewport.max_depth,
	};
	self.vk.vkCmdSetViewportWithCount(self.command_buffer, 1, &vk_viewport);
}

fn void CommandBuffer.set_scissor(&self, Scissor scissor)
{
	VkRect2D vk_scissor = { .offset = { scissor.x, scissor.y }, .extent = { scissor.width, scissor.height } };
	self.vk.vkCmdSetScissorWithCount(self.command_buffer, 1, &vk_scissor);
}

<*
 Depth and stencil state is dynamic. Every render pass starts with both tests off.
*>
fn void CommandBuffer.set_depth_stencil(&self, DepthStencilState state)
{
	VkFns* vk = self.vk;
	VkCommandBuffer cb = self.command_buffer;
	vk.vkCmdSetDepthTestEnable(cb, state.depth_test ? VK_TRUE : VK_FALSE);
	if (state.depth_test)
	{
		vk.vkCmdSetDepthWriteEnable(cb, state.depth_write ? VK_TRUE : VK_FALSE);
		vk.vkCmdSetDepthCompareOp(cb, vk_compare_op(state.depth_compare));
	}
	vk.vkCmdSetStencilTestEnable(cb, state.stencil_test ? VK_TRUE : VK_FALSE);
	if (!state.stencil_test) return;
	vk.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, (VkStencilOp)state.front.fail.ordinal, (VkStencilOp)state.front.pass.ordinal,
		(VkStencilOp)state.front.depth_fail.ordinal, vk_compare_op(state.front.compare.or(ALWAYS)));
	vk.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, (VkStencilOp)state.back.fail.ordinal, (VkStencilOp)state.back.pass.ordinal,
		(VkStencilOp)state.back.depth_fail.ordinal, vk_compare_op(state.back.compare.or(ALWAYS)));
	vk.vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, state.stencil_read_mask ?: 0xff);
	vk.vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, state.stencil_write_mask ?: 0xff);
	vk.vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, state.front.reference);
	vk.vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, state.back.reference);
}

fn VkRenderingAttachmentInfo rendering_attachment(RenderView* view, LoadOp load, StoreOp store) @private => {
	.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	.imageView = view ? view.view : 0,
	.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	.loadOp = (VkAttachmentLoadOp)load.ordinal,
	.storeOp = (VkAttachmentStoreOp)store.ordinal,
};

<*
 Begins rendering to the attachments. The render area is the extent of the first attachment, and starts with a full viewport and scissor
 and with depth and stencil off. No pipeline is bound: bind one inside the pass, also in a resumed segment, and bind a compute pipeline
 again before dispatching after the pass. (Vulkan keeps its bindings across passes, but that is not the API.) A render pass needs at least one attachment. `suspending` and `resuming` split one pass over several
 command buffers: the first segment loads, the last stores, and submitting the whole chain in order in one batch is the caller's job.
*>
fn void CommandBuffer.begin_render_pass(&self, RenderingDesc desc, RenderingFlags flags = {})
{
	assert(desc.colors.len <= MAX_COLOR_ATTACHMENTS);
	RenderView* area_view = desc.colors.len ? desc.colors[0].render_view : (desc.depth.render_view ?: desc.stencil.render_view);
	assert(area_view, "begin_render_pass requires an attachment to determine the render area");

	VkRenderingAttachmentInfo[MAX_COLOR_ATTACHMENTS] color_attachments;
	foreach (index, &attachment : desc.colors)
	{
		assert(attachment.render_view);
		color_attachments[index] = rendering_attachment(attachment.render_view, attachment.load, attachment.store);
		color_attachments[index].clearValue.color.float32 = { attachment.clear.x, attachment.clear.y, attachment.clear.z, attachment.clear.w };
	}
	VkRenderingAttachmentInfo depth_attachment = rendering_attachment(desc.depth.render_view, desc.depth.load, desc.depth.store);
	depth_attachment.clearValue.depthStencil.depth = desc.depth.clear;
	VkRenderingAttachmentInfo stencil_attachment = rendering_attachment(desc.stencil.render_view, desc.stencil.load, desc.stencil.store);
	stencil_attachment.clearValue.depthStencil.stencil = desc.stencil.clear;
	VkRenderingInfo rendering_info = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.flags = (VkRenderingFlags)(uint)flags,
		.renderArea = { .extent = { area_view.width, area_view.height } },
		.layerCount = 1,
		.colorAttachmentCount = (uint)desc.colors.len,
		.pColorAttachments = &color_attachments[0],
		.pDepthAttachment = desc.depth.render_view ? &depth_attachment : null,
		.pStencilAttachment = desc.stencil.render_view ? &stencil_attachment : null,
	};
	self.vk.vkCmdBeginRendering(self.command_buffer, &rendering_info);

	self.set_viewport({ .width = (float)area_view.width, .height = (float)area_view.height, .min_depth = 0.0f, .max_depth = 1.0f });
	self.set_scissor({ .width = area_view.width, .height = area_view.height });
	self.set_depth_stencil({});
}

fn void CommandBuffer.end_render_pass(&self)
{
	self.vk.vkCmdEndRendering(self.command_buffer);
}

<*
 Binds a graphics or mesh pipeline inside a render pass, a compute pipeline outside one.
*>
fn void CommandBuffer.bind_pso(&self, Pso* pso)
{
	assert(pso);
	self.vk.vkCmdBindPipeline(self.command_buffer, pso.bind_point, pso.pipeline);
}

// ---- Draws and dispatches -------------------------------------------------------------------------------------------------------

<*
 The root: the address of a 16 byte aligned structure that the shaders read, pushed as eight bytes of push data. Null for shaders
 without root data.
*>
fn void emit_root_pointer(CommandBuffer* commands, void* root) @private
{
	if (!root) return;
	assert(((uptr)root & 15) == 0, "a root pointer must be 16 byte aligned");
	VkPushDataInfoEXT info = { .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT, .data = { .address = &root, .size = void*::size } };
	commands.vk.vkCmdPushDataEXT(commands.command_buffer, &info);
}

fn void CommandBuffer.draw(&self, void* root, uint vertex_count, uint instance_count = 1, uint first_vertex = 0, uint first_instance = 0)
{
	emit_root_pointer(self, root);
	self.vk.vkCmdDraw(self.command_buffer, vertex_count, instance_count, first_vertex, first_instance);
}

fn void bind_indices(CommandBuffer* commands, GpuRange indices, IndexType type) @private
{
	if (commands.device.buffer_map)
	{
		ulong offset;
		VkBuffer buffer = commands.resolve(indices, &offset);
		commands.vk.vkCmdBindIndexBuffer2(commands.command_buffer, buffer, offset, indices.size, (VkIndexType)type.ordinal);
		return;
	}
	VkBindIndexBuffer3InfoKHR bind_info = {
		.sType = VK_STRUCTURE_TYPE_BIND_INDEX_BUFFER_3_INFO_KHR,
		.addressRange = address_range(indices),
		.addressFlags = ADDRESS_FLAGS,
		.indexType = (VkIndexType)type.ordinal,
	};
	commands.vk.vkCmdBindIndexBuffer3KHR(commands.command_buffer, &bind_info);
}

fn VkDrawIndirect2InfoKHR draw_indirect_info(GpuRange arguments, uint draw_count, uint stride, uint default_stride) @private => {
	.sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
	.addressRange = { .address = (VkDeviceAddress)(uptr)arguments.gpu, .size = arguments.size, .stride = stride ?: default_stride },
	.addressFlags = ADDRESS_FLAGS,
	.drawCount = draw_count,
};

fn void CommandBuffer.draw_indexed(&self, void* root, GpuRange indices, IndexType type, uint index_count, uint instance_count = 1,
	uint first_index = 0, int vertex_offset = 0, uint first_instance = 0)
{
	emit_root_pointer(self, root);
	bind_indices(self, indices, type);
	self.vk.vkCmdDrawIndexed(self.command_buffer, index_count, instance_count, first_index, vertex_offset, first_instance);
}

fn void CommandBuffer.draw_indirect(&self, void* root, GpuRange arguments, uint draw_count = 1, uint stride = 0)
{
	emit_root_pointer(self, root);
	if (self.device.buffer_map)
	{
		ulong offset;
		VkBuffer buffer = self.resolve(arguments, &offset);
		self.vk.vkCmdDrawIndirect(self.command_buffer, buffer, offset, draw_count, stride ?: (uint)VkDrawIndirectCommand::size);
		return;
	}
	VkDrawIndirect2InfoKHR info = draw_indirect_info(arguments, draw_count, stride, VkDrawIndirectCommand::size);
	self.vk.vkCmdDrawIndirect2KHR(self.command_buffer, &info);
}

fn void CommandBuffer.draw_indexed_indirect(&self, void* root, GpuRange indices, IndexType type, GpuRange arguments, uint draw_count = 1, uint stride = 0)
{
	emit_root_pointer(self, root);
	bind_indices(self, indices, type);
	if (self.device.buffer_map)
	{
		ulong offset;
		VkBuffer buffer = self.resolve(arguments, &offset);
		self.vk.vkCmdDrawIndexedIndirect(self.command_buffer, buffer, offset, draw_count, stride ?: (uint)VkDrawIndexedIndirectCommand::size);
		return;
	}
	VkDrawIndirect2InfoKHR info = draw_indirect_info(arguments, draw_count, stride, VkDrawIndexedIndirectCommand::size);
	self.vk.vkCmdDrawIndexedIndirect2KHR(self.command_buffer, &info);
}

fn void CommandBuffer.dispatch(&self, void* root, Uint3 group_count)
{
	emit_root_pointer(self, root);
	self.vk.vkCmdDispatch(self.command_buffer, group_count.x, group_count.y, group_count.z);
}

fn void CommandBuffer.dispatch_indirect(&self, void* root, GpuRange arguments)
{
	emit_root_pointer(self, root);
	if (self.device.buffer_map)
	{
		ulong offset;
		VkBuffer buffer = self.resolve(arguments, &offset);
		self.vk.vkCmdDispatchIndirect(self.command_buffer, buffer, offset);
		return;
	}
	VkDispatchIndirect2InfoKHR info = {
		.sType = VK_STRUCTURE_TYPE_DISPATCH_INDIRECT_2_INFO_KHR,
		.addressRange = address_range(arguments),
		.addressFlags = ADDRESS_FLAGS,
	};
	self.vk.vkCmdDispatchIndirect2KHR(self.command_buffer, &info);
}

fn void CommandBuffer.draw_meshlets(&self, void* root, Uint3 group_count)
{
	emit_root_pointer(self, root);
	self.vk.vkCmdDrawMeshTasksEXT(self.command_buffer, group_count.x, group_count.y, group_count.z);
}

<*
 The arguments are VkDrawMeshTasksIndirectCommandEXT records. Always available: the extensions the device needs provide it.
*>
fn void CommandBuffer.draw_meshlets_indirect(&self, void* root, GpuRange arguments, uint draw_count = 1, uint stride = 0)
{
	emit_root_pointer(self, root);
	if (self.device.buffer_map)
	{
		ulong offset;
		VkBuffer buffer = self.resolve(arguments, &offset);
		self.vk.vkCmdDrawMeshTasksIndirectEXT(self.command_buffer, buffer, offset, draw_count, stride ?: (uint)VkDrawMeshTasksIndirectCommandEXT::size);
		return;
	}
	VkDrawIndirect2InfoKHR info = draw_indirect_info(arguments, draw_count, stride, VkDrawMeshTasksIndirectCommandEXT::size);
	self.vk.vkCmdDrawMeshTasksIndirect2EXT(self.command_buffer, &info);
}
