const threadsPerThreadgroup = 32;

const originalBufferBindingNum = 0;
const outputBufferBindingNum = 1;
const uniformsBufferBindingNum = 2;

function init() {
    if (!navigator.gpu) {
        document.body.className = 'error';
        return;
    }

    const slider = document.getElementById("radiusSlider");
    const button = document.getElementById("blurButton");

    button.onclick = async () => {
        button.disabled = true;
        slider.disabled = true;
        await computeBlur(4);
        button.disabled = false;
        slider.disabled = false;
    };
}

let uniformsCache = new Map();

async function computeBlur(radius) {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const canvas = document.querySelector("canvas");
    const context2d = canvas.getContext("2d");

    /* Image */
    const image = new Image();
    const imageLoadPromise = new Promise(resolve => { 
        image.onload = () => resolve(); 
        image.src = "resources/safari-opaque.png"
    });
    await Promise.resolve(imageLoadPromise);

    canvas.height = image.height;
    canvas.width = image.width;

    context2d.drawImage(image, 0, 0);

    const originalData = context2d.getImageData(0, 0, image.width, image.height);
    const imageLength = originalData.data.length;

    const [originalBuffer, originalArrayBuffer] = device.createBufferMapped({ 
        size: imageLength, 
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.MAP_READ 
    });
    const imageWriteArray = new Uint8ClampedArray(originalArrayBuffer);
    imageWriteArray.set(originalData.data);
    originalBuffer.unmap();

    // Create output buffer
    const outputBuffer = device.createBuffer({ size: imageLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.MAP_READ });

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
            binding: outputBufferBindingNum,
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
                size: imageLength
            }
        }, {
            binding: outputBufferBindingNum,
            resource: {
                buffer: outputBuffer,
                size: imageLength
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
    let passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(horizontalPipeline);
    const numXGroups = Math.ceil(image.width / threadsPerThreadgroup);
    passEncoder.dispatch(numXGroups, image.height, 1);

    // Run vertical pass back to originalBuffer
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(verticalPipeline);
    const numYGroups = Math.ceil(image.height / threadsPerThreadgroup);
    passEncoder.dispatch(image.width, numYGroups, 1);
    passEncoder.endPass();

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

	for (let i = -radius; i <= radius; ++i)
	{
        const weight = Math.exp(-i * i / twoSigma2);
        weightSum += weight;
        if (i >= 0)
            weights.push(weight);
    }

	for (let i = 1; i < weights.length; ++i)
	{
		weights[i] /= weightSum;
	}

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
    // Because little-endian? ¯\_(ツ)_/¯
    return r + (g << 8) + (b << 16) + (a << 24);
}

uint weightedColor(uint startColor, float weight)
{
    uint r = uint(float(getR(startColor)) * weight);
    uint g = uint(float(getG(startColor)) * weight);
    uint b = uint(float(getB(startColor)) * weight);
    uint a = uint(float(getA(startColor)) * weight);

    return makeRGBA(r, g, b, a);
}

uint verticallyOffsetIndex(uint index, int offset)
{
    int realOffset = offset * ${image.width};

    if (int(index) + realOffset < 0)
        return 0;
    
    return uint(int(index) + realOffset);
}

[numthreads(${threadsPerThreadgroup}, 1, 1)]
compute void horizontal(constant uint[] origBuffer : register(u${originalBufferBindingNum}),
                        device uint[] outputBuffer : register(u${outputBufferBindingNum}),
                        constant float[] uniforms : register(b${uniformsBufferBindingNum}),
                        float3 dispatchThreadID : SV_DispatchThreadID)
{
    int radius = int(uniforms[0]);
    uint globalIndex = uint(dispatchThreadID.y) * ${image.width} + uint(dispatchThreadID.x);

    uint color = 0;

    for (int i = -radius; i <= radius; ++i) {
        uint startColor = origBuffer[uint(int(globalIndex) + i)];
        float weight = uniforms[uint(abs(i) + 1)];
        color += weightedColor(startColor, weight);
    }

    outputBuffer[globalIndex] = color;
}

[numthreads(1, ${threadsPerThreadgroup}, 1)]
compute void vertical(device uint[] origBuffer : register(u${originalBufferBindingNum}),
                        constant uint[] middleBuffer : register(u${outputBufferBindingNum}),
                        constant float[] uniforms : register(b${uniformsBufferBindingNum}),
                        float3 dispatchThreadID : SV_DispatchThreadID)
{
    int radius = int(uniforms[0]);
    uint globalIndex = uint(dispatchThreadID.x) * ${image.height} + uint(dispatchThreadID.y);

    uint color = 0;

    for (int i = -radius; i <= radius; ++i) {
        uint startColor = middleBuffer[verticallyOffsetIndex(globalIndex, i)];
        float weight = uniforms[uint(abs(i) + 1)];
        color += weightedColor(startColor, weight);
    }

    origBuffer[globalIndex] = color;
}
`;
}