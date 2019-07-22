const threadsPerThreadgroup = 32;

const originalBufferBindingNum = 0;
const storageBufferBindingNum = 1;
const uniformsBufferBindingNum = 2;

async function init() {
    if (!navigator.gpu) {
        document.body.className = "error";
        return;
    }

    const slider = document.querySelector("input");
    const canvas = document.querySelector("canvas");
    const context2d = canvas.getContext("2d");

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const image = await loadImage(canvas, context2d);

    slider.onchange = async () => {
        slider.disabled = true;
        await computeBlur(slider.value, image, context2d, device);
        slider.disabled = false;
    };
}

async function loadImage(canvas, context2d) {
    /* Image */
    const image = new Image();
    const imageLoadPromise = new Promise(resolve => { 
        image.onload = () => resolve(); 
        image.src = "resources/safari-alpha.png"
    });
    await Promise.resolve(imageLoadPromise);

    canvas.height = image.height;
    canvas.width = image.width;

    context2d.drawImage(image, 0, 0);

    return image;
}

let uniformsCache = new Map();
let originalData;
let storageBuffer;

async function computeBlur(radius, image, context2d, device) {
    if (radius == 0) {
        context2d.drawImage(image, 0, 0);
        return;
    }

    if (originalData === undefined)
        originalData = context2d.getImageData(0, 0, image.width, image.height);
    const imageSize = originalData.data.length;

    const [originalBuffer, originalArrayBuffer] = device.createBufferMapped({ 
        size: imageSize, 
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.MAP_READ 
    });
    const imageWriteArray = new Uint8ClampedArray(originalArrayBuffer);
    imageWriteArray.set(originalData.data);
    originalBuffer.unmap();

    // Create storage buffer
    if (storageBuffer === undefined) 
        storageBuffer = device.createBuffer({ size: imageSize, usage: GPUBufferUsage.STORAGE });

    // Create uniforms buffer
    let uniforms = uniformsCache.get(radius);
    if (uniforms === undefined) {
        uniforms = calculateWeights(radius);
        uniformsCache.set(radius, uniforms);
    }
    const uniformsBufferSize = uniforms.length * Float32Array.BYTES_PER_ELEMENT;

    const [uniformsBuffer, uniformsArrayBuffer] = device.createBufferMapped({ size: uniformsBufferSize, usage: GPUBufferUsage.UNIFORM });
    const uniformsWriteArray = new Float32Array(uniformsArrayBuffer);
    uniformsWriteArray.set(uniforms);
    uniformsBuffer.unmap();

    // Bind buffers to kernel   
    const bindGroupLayout = device.createBindGroupLayout({
        bindings: [{
            binding: originalBufferBindingNum,
            visibility: GPUShaderStageBit.COMPUTE,
            type: "storage-buffer"
        }, {
            binding: storageBufferBindingNum,
            visibility: GPUShaderStageBit.COMPUTE,
            type: "storage-buffer"
        }, {
            binding: uniformsBufferBindingNum,
            visibility: GPUShaderStageBit.COMPUTE,
            type: "uniform-buffer"
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        bindings: [{
            binding: originalBufferBindingNum,
            resource: {
                buffer: originalBuffer,
                size: imageSize
            }
        }, {
            binding: storageBufferBindingNum,
            resource: {
                buffer: storageBuffer,
                size: imageSize
            }
        }, {
            binding: uniformsBufferBindingNum,
            resource: {
                buffer: uniformsBuffer,
                size: uniformsBufferSize
            }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // Set up pipelines
    const shaderModule = device.createShaderModule({ code: createShaderCode(image), isWHLSL: true });

    device.pushErrorScope("validation");

    const horizontalPipeline = device.createComputePipeline({ 
        layout: pipelineLayout, 
        computeStage: {
            module: shaderModule,
            entryPoint: "horizontal"
        }
    });

    device.popErrorScope().then(error => { if (error) console.log(error.message); });

    device.pushErrorScope("validation");

    const verticalPipeline = device.createComputePipeline({
        layout: pipelineLayout,
        computeStage: {
            module: shaderModule,
            entryPoint: "vertical"
        }
    });

    device.popErrorScope().then(error => { if (error) console.log(error.message); });

    // Run horizontal pass first
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(horizontalPipeline);
    const numXGroups = Math.ceil(image.width / threadsPerThreadgroup);
    passEncoder.dispatch(numXGroups, image.height, 1);
    passEncoder.endPass();

    // Run vertical pass back to originalBuffer
    const verticalPassEncoder = commandEncoder.beginComputePass();
    verticalPassEncoder.setBindGroup(0, bindGroup);
    verticalPassEncoder.setPipeline(verticalPipeline);
    const numYGroups = Math.ceil(image.height / threadsPerThreadgroup);
    verticalPassEncoder.dispatch(image.width, numYGroups, 1);
    verticalPassEncoder.endPass();

    device.getQueue().submit([commandEncoder.finish()]);

    // Draw originalBuffer as imageData back into context2d
    const resultArrayBuffer = await originalBuffer.mapReadAsync();
    const resultArray = new Uint8ClampedArray(resultArrayBuffer);
    context2d.putImageData(new ImageData(resultArray, image.width, image.height), 0, 0);
}

window.addEventListener("load", init);

/* Helpers */

function calculateWeights(radius)
{
    const sigma = radius / 2.0;
	const twoSigma2 = 2.0 * sigma * sigma;

    let weights = [radius];
	let weightSum = 0;

	for (let i = 0; i <= radius; ++i) {
        const weight = Math.exp(-i * i / twoSigma2);
        weights.push(weight);
        weightSum += (i == 0) ? weight : weight * 2;
    }

    // Compensate for loss in brightness
    const brightnessScale =  1 - (0.1 / 32.0) * radius;
    weightSum *= brightnessScale;
	for (let i = 1; i < weights.length; ++i)
		weights[i] /= weightSum;

	return weights;
}

const byteMask = (1 << 8) - 1;

function createShaderCode(image) {
    return `
uint getR(uint rgba)
{
    return rgba & ${byteMask};
}

uint getG(uint rgba)
{
    return (rgba >> 8) & ${byteMask};
}

uint getB(uint rgba)
{
    return (rgba >> 16) & ${byteMask};
}

uint getA(uint rgba)
{
    return (rgba >> 24) & ${byteMask};
}

uint makeRGBA(uint r, uint g, uint b, uint a)
{
    return r + (g << 8) + (b << 16) + (a << 24);
}

void accumulateChannels(thread uint[] channels, uint startColor, float weight)
{
    channels[0] += uint(float(getR(startColor)) * weight);
    channels[1] += uint(float(getG(startColor)) * weight);
    channels[2] += uint(float(getB(startColor)) * weight);
    channels[3] += uint(float(getA(startColor)) * weight);

    // Compensate for brightness-adjusted weights.
    if (channels[0] > 255)
        channels[0] = 255;

    if (channels[1] > 255)
        channels[1] = 255;

    if (channels[2] > 255)
        channels[2] = 255;

    if (channels[3] > 255)
        channels[3] = 255;
}

uint horizontallyOffsetIndex(uint index, int offset, int rowStart, int rowEnd)
{
    int offsetIndex = int(index) + offset;

    if (offsetIndex < rowStart || offsetIndex >= rowEnd)
        return index;
    
    return uint(offsetIndex);
}

uint verticallyOffsetIndex(uint index, int offset, uint length)
{
    int realOffset = offset * ${image.width};
    int offsetIndex = int(index) + realOffset;

    if (offsetIndex < 0 || offsetIndex >= int(length))
        return index;
    
    return uint(offsetIndex);
}

[numthreads(${threadsPerThreadgroup}, 1, 1)]
compute void horizontal(constant uint[] origBuffer : register(u${originalBufferBindingNum}),
                        device uint[] storageBuffer : register(u${storageBufferBindingNum}),
                        constant float[] uniforms : register(b${uniformsBufferBindingNum}),
                        float3 dispatchThreadID : SV_DispatchThreadID)
{
    int radius = int(uniforms[0]);
    int rowStart = ${image.width} * int(dispatchThreadID.y);
    int rowEnd = ${image.width} * (1 + int(dispatchThreadID.y));
    uint globalIndex = uint(rowStart) + uint(dispatchThreadID.x);

    uint[4] channels;

    for (int i = -radius; i <= radius; ++i) {
        uint startColor = origBuffer[horizontallyOffsetIndex(globalIndex, i, rowStart, rowEnd)];
        float weight = uniforms[uint(abs(i) + 1)];
        accumulateChannels(@channels, startColor, weight);
    }

    storageBuffer[globalIndex] = makeRGBA(channels[0], channels[1], channels[2], channels[3]);
}

[numthreads(1, ${threadsPerThreadgroup}, 1)]
compute void vertical(device uint[] origBuffer : register(u${originalBufferBindingNum}),
                        constant uint[] middleBuffer : register(u${storageBufferBindingNum}),
                        constant float[] uniforms : register(b${uniformsBufferBindingNum}),
                        float3 dispatchThreadID : SV_DispatchThreadID)
{
    int radius = int(uniforms[0]);
    uint globalIndex = uint(dispatchThreadID.x) * ${image.height} + uint(dispatchThreadID.y);

    uint[4] channels;

    for (int i = -radius; i <= radius; ++i) {
        uint startColor = middleBuffer[verticallyOffsetIndex(globalIndex, i, middleBuffer.length)];
        float weight = uniforms[uint(abs(i) + 1)];
        accumulateChannels(@channels, startColor, weight);
    }

    origBuffer[globalIndex] = makeRGBA(channels[0], channels[1], channels[2], channels[3]);
}
`;
}