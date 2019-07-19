const threadsPerThreadgroup = 32;
const blurRadius = 2;
const byteMask = (1 << 8) - 1;

const originalBufferBindingNum = 0;
const outputBufferBindingNum = 1;

const shaderUtils = `
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
`;

async function computeBlur() {
    if (!navigator.gpu) {
        document.body.className = 'error';
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const canvas = document.querySelector("canvas");
    const context2d = canvas.getContext("2d");

    /* Image */
    const image = new Image();
    const imageLoadPromise = new Promise(resolve => { 
        image.onload = () => resolve(); 
        image.src = "resources/blue-checkered.png"
    });
    await Promise.resolve(imageLoadPromise);

    canvas.height = image.height;
    canvas.width = image.width;

    context2d.drawImage(image, 0, 0);

    const originalData = context2d.getImageData(0, 0, image.width, image.height);
    const imageLength = originalData.data.length;

    const [originalBuffer, originalArrayBuffer] = device.createBufferMapped({ size: imageLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.MAP_READ });

    const imageWriteArray = new Uint8ClampedArray(originalArrayBuffer);
    imageWriteArray.set(originalData.data);
    originalBuffer.unmap();

    // Create output buffer
    const outputBuffer = device.createBuffer({ size: imageLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.MAP_READ });

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
        }]
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // Set up pipelines
    const horizontalModule = device.createShaderModule({ 
        code: shaderUtils + `
[numthreads(${threadsPerThreadgroup}, 1, 1)]
compute void horizontal(device uint[] origBuffer : register(u${originalBufferBindingNum}),
                        device uint[] outputBuffer : register(u${outputBufferBindingNum}),
                        float3 dispatchThreadID : SV_DispatchThreadID)
{
    float[5] weights;
    weights[0] = 0.2270270270;
    weights[1] = 0.1945945946;
    weights[2] = 0.1216216216;
    weights[3] = 0.0540540541;
    weights[4] = 0.0162162162;

    uint globalIndex = uint(dispatchThreadID.y) * ${image.width} + uint(dispatchThreadID.x);

    uint originalRGBA = origBuffer[globalIndex];
    uint r = 0;
    uint g = 0;
    uint b = 0;
    uint a = 0;

    for (int i = -${blurRadius}; i <= ${blurRadius}; ++i) {
        uint originalRGBA = origBuffer[uint(int(globalIndex) + i)];
        r += uint(float(getR(originalRGBA)) * weights[uint(i + ${blurRadius})]);
        g += uint(float(getG(originalRGBA)) * weights[uint(i + ${blurRadius})]);
        b += uint(float(getB(originalRGBA)) * weights[uint(i + ${blurRadius})]);
        a += uint(float(getA(originalRGBA)) * weights[uint(i + ${blurRadius})]);
    }

    outputBuffer[globalIndex] = makeRGBA(r, g, b, a);
}`, 
        isWHLSL: true
    });

    device.pushErrorScope("validation");

    const horizontalPipeline = device.createComputePipeline({ 
        layout: pipelineLayout, 
        computeStage: {
            module: horizontalModule,
            entryPoint: "horizontal"
        }
    });

    device.popErrorScope().then(error => { if (error) console.log(error.message); });

    const verticalModule = device.createShaderModule({
        code: shaderUtils + `
[numthreads(1, 1, 1)]
compute void vertical() {}
`,
        isWHLSL: true
    });

    const verticalPipeline = device.createComputePipeline({
        layout: pipelineLayout,
        computeStage: {
            module: verticalModule,
            entryPoint: "vertical"
        }
    });

    // Run horizontal pass first

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(horizontalPipeline);
    const numXGroups = Math.ceil(image.width / threadsPerThreadgroup);
    passEncoder.dispatch(numXGroups, image.height, 1);
    //passEncoder.endPass();

    // Run vertical pass back to originalBuffer

    passEncoder.setPipeline(verticalPipeline);
    const numYGroups = Math.ceil(image.height / threadsPerThreadgroup);
    passEncoder.dispatch(image.width, numYGroups, 1);
    passEncoder.endPass();

    device.getQueue().submit([commandEncoder.finish()]);

    // Draw originalBuffer as imageData back into context2d
    const resultArrayBuffer = await outputBuffer.mapReadAsync();

    const resultArray = new Uint8ClampedArray(resultArrayBuffer);
    context2d.putImageData(new ImageData(resultArray, image.width, image.height), 0, 0);
}

window.addEventListener("load", computeBlur);

const verticalShader = `
const float[] offset = [0.0, 1.0, 2.0, 3.0, 4.0];
const float[] weight = [
  0.2270270270, 0.1945945946, 0.1216216216,
  0.0540540541, 0.0162162162
];

[numthreads(1, N, 1)]
void VertBlurCS(int3 groupThreadID : SV_GroupThreadID,
				int3 dispatchThreadID : SV_DispatchThreadID)
{
	// Put in an array for each indexing.
	float weights[11] = { w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 };

	//
	// Fill local thread storage to reduce bandwidth.  To blur 
	// N pixels, we will need to load N + 2*BlurRadius pixels
	// due to the blur radius.
	//
	
	// This thread group runs N threads.  To get the extra 2*BlurRadius pixels, 
	// have 2*BlurRadius threads sample an extra pixel.
	if(groupThreadID.y < gBlurRadius)
	{
		// Clamp out of bound samples that occur at image borders.
		int y = max(dispatchThreadID.y - gBlurRadius, 0);
		gCache[groupThreadID.y] = gInput[int2(dispatchThreadID.x, y)];
	}
	if(groupThreadID.y >= N-gBlurRadius)
	{
		// Clamp out of bound samples that occur at image borders.
		int y = min(dispatchThreadID.y + gBlurRadius, gInput.Length.y-1);
		gCache[groupThreadID.y+2*gBlurRadius] = gInput[int2(dispatchThreadID.x, y)];
	}
	
	// Clamp out of bound samples that occur at image borders.
	gCache[groupThreadID.y+gBlurRadius] = gInput[min(dispatchThreadID.xy, gInput.Length.xy-1)];


	// Wait for all threads to finish.
	GroupMemoryBarrierWithGroupSync();
	
	//
	// Now blur each pixel.
	//

	float4 blurColor = float4(0, 0, 0, 0);
	
	for(int i = -gBlurRadius; i <= gBlurRadius; ++i)
	{
		int k = groupThreadID.y + gBlurRadius + i;
		
		blurColor += weights[i+gBlurRadius]*gCache[k];
	}
	
	gOutput[dispatchThreadID.xy] = blurColor;
}
`;

const s = `

// uint2 globalIndex = uint2(uint(dispatchThreadID.x), uint(dispatchThreadID.y));

threadgroup uint[34] gCache;

if (localIndex < ${blurRadius}) {
    uint x = uint(max(int(localIndex) - ${blurRadius}, 0));
    gCache[localIndex] = origBuffer[globalIndex.y * image.width + x];
}

if (localIndex >= nMinusBlurRadius) {
    uint x = min(globalIndex.x + ${blurRadius}, uint(image.width - 1));
    gCache[localIndex] = origBuffer[globalIndex.y * image.width + x];
}

void horizontal(int3 groupThreadID : SV_GroupThreadID,
				int3 dispatchThreadID : SV_DispatchThreadID)
{
	// Put in an array for each indexing.
	float weights[11] = { w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 };

	//
	// Fill local thread storage to reduce bandwidth.  To blur 
	// N pixels, we will need to load N + 2*BlurRadius pixels
	// due to the blur radius.
	//
	
	// This thread group runs N threads.  To get the extra 2*BlurRadius pixels, 
	// have 2*BlurRadius threads sample an extra pixel.
	if(groupThreadID.x < gBlurRadius)
	{
		// Clamp out of bound samples that occur at image borders.
		int x = max(dispatchThreadID.x - gBlurRadius, 0);
		gCache[groupThreadID.x] = gInput[int2(x, dispatchThreadID.y)];
	}
	if(groupThreadID.x >= N-gBlurRadius)
	{
		// Clamp out of bound samples that occur at image borders.
		int x = min(dispatchThreadID.x + gBlurRadius, gInput.Length.x-1);
		gCache[groupThreadID.x+2*gBlurRadius] = gInput[int2(x, dispatchThreadID.y)];
	}

	// Clamp out of bound samples that occur at image borders.
	gCache[groupThreadID.x+gBlurRadius] = gInput[min(dispatchThreadID.xy, gInput.Length.xy-1)];

	// Wait for all threads to finish.
	GroupMemoryBarrierWithGroupSync();
	
	//
	// Now blur each pixel.
	//

	float4 blurColor = float4(0, 0, 0, 0);
	
	for(int i = -gBlurRadius; i <= gBlurRadius; ++i)
	{
		int k = groupThreadID.x + gBlurRadius + i;
		
		blurColor += weights[i+gBlurRadius]*gCache[k];
	}
	
	gOutput[dispatchThreadID.xy] = blurColor;
}`;