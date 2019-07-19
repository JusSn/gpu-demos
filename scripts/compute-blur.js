const threadsPerThreadgroup = 32;

const originalBufferBindingNum = 0;
const outputBufferBindingNum = 1;
const weightsBufferBindingNum = 2;

const weights = [
    0.2270270270,
    0.1945945946,
    0.1216216216,
    0.0540540541,
    0.0162162162,
];
const weightsBufferSize = weights.length * Float32Array.BYTES_PER_ELEMENT;

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
    const [weightsBuffer, weightsArrayBuffer] = device.createBufferMapped({ size: weightsBufferSize, usage: GPUBufferUsage.UNIFORM });
    const weightsWriteArray = new Float32Array(weightsArrayBuffer);
    weightsWriteArray.set(weights);
    weightsBuffer.unmap();

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
            binding: weightsBufferBindingNum,
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
            binding: weightsBufferBindingNum,
            resource: {
                buffer: weightsBuffer,
                size: weightsBufferSize
            }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // Set up pipelines
    const shaderModule = device.createShaderModule({ code: createShaderCode(image, radius), isWHLSL: true });

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

/* Shaders */

const byteMask = (1 << 8) - 1;

function createShaderCode(image, radius) {
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
                        constant float[] weights : register(b${weightsBufferBindingNum}),
                        float3 dispatchThreadID : SV_DispatchThreadID)
{
    uint globalIndex = uint(dispatchThreadID.y) * ${image.width} + uint(dispatchThreadID.x);

    uint r = 0;
    uint g = 0;
    uint b = 0;
    uint a = 0;

    for (int i = -${radius}; i <= ${radius}; ++i) {
        uint startColor = origBuffer[uint(int(globalIndex) + i)];
        float weight = weights[uint(abs(i))];
        r += uint(float(getR(startColor)) * weight);
        g += uint(float(getG(startColor)) * weight);
        b += uint(float(getB(startColor)) * weight);
        a += uint(float(getA(startColor)) * weight);
    }

    outputBuffer[globalIndex] = makeRGBA(r, g, b, a);
}

[numthreads(1, ${threadsPerThreadgroup}, 1)]
compute void vertical(device uint[] origBuffer : register(u${originalBufferBindingNum}),
                        constant uint[] middleBuffer : register(u${outputBufferBindingNum}),
                        constant float[] weights : register(b${weightsBufferBindingNum}),
                        float3 dispatchThreadID : SV_DispatchThreadID)
{
    uint globalIndex = uint(dispatchThreadID.x) * ${image.height} + uint(dispatchThreadID.y);

    uint r = 0;
    uint g = 0;
    uint b = 0;
    uint a = 0;

    for (int i = -${radius}; i <= ${radius}; ++i) {
        uint startColor = middleBuffer[verticallyOffsetIndex(globalIndex, i)];
        float weight = weights[uint(abs(i))];
        r += uint(float(getR(startColor)) * weight);
        g += uint(float(getG(startColor)) * weight);
        b += uint(float(getB(startColor)) * weight);
        a += uint(float(getA(startColor)) * weight);
    }

    origBuffer[globalIndex] = makeRGBA(r, g, b, a);
}
`;
}