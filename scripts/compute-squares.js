const MAX_THREAD_NUM = 1024;
const MAX_GROUP_NUM = 2048;
const MAX_ORIGINAL_VALUE = 94906265; // Largest safe square root in JS.

let logElement;
let selectBox;

let device;
const dataBinding = 0;
const bindGroupIndex = 0;

const main = async () => {
  // Selector setup
  selectBox = document.getElementById('selectBox');
  const maxNumElementsIndex = Math.log2(MAX_THREAD_NUM * MAX_GROUP_NUM) - 9;
  for (let i = 0; i < maxNumElementsIndex; i++) {
    const option = document.createElement('option');
    option.text = '' + getLength(i);
    selectBox.add(option);
  }
  selectBox.selectedIndex = 7;
  selectBox.addEventListener('change', () => {
    logElement.innerText = '';
    selectBox.disabled = true;
    requestAnimationFrame(() => compute());
  });
  selectBox.disabled = true;

  // Div setup
  logElement = document.getElementById('log');

  // Create GPUDevice
  const adapter = await gpu.requestAdapter();
  device = await adapter.requestDevice();

  if (!device) {
    document.body.className = 'error';
    return;
  }

  compute();
};

const compute = async () => {
  const length = getLength(selectBox.selectedIndex);
  console.log(`square test: ${length}`);
  const arr = new Uint32Array(length);
  resetData(arr);

  await computeCPU(arr.slice(0));
  await computeGPU(arr.slice(0));

  selectBox.disabled = false;
  console.log(`---`);
};

const computeCPU = async (arr) => {
  const now = performance.now();
  arr.forEach((value, index) => {
    arr[index] = value * value;
  });
  log(`CPU square time: ${Math.round(performance.now() - now)} ms`)
  console.log(arr);
};

const computeGPU = async (arr) => {
  const now = performance.now();

  // const threadgroupsPerGrid = Math.max(1, length / MAX_THREAD_NUM);

  const shaderModule = createComputeShader(arr.length);
  const pipeline = device.createComputePipeline({ 
    computeStage: { module: shaderModule, entryPoint: "squares_main" } 
  });
  
  const dataBuffer = device.createBuffer({ 
    size: arr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.TRANSFER_DST | GPUBufferUsage.MAP_READ
  });
  dataBuffer.setSubData(0, arr.buffer);

  const dataBindGroupLayout = device.createBindGroupLayout({ 
    bindings: [{ binding: dataBinding, visibility: GPUShaderStageBit.COMPUTE, type: "storage-buffer" }]
  });

  const dataBindGroup = device.createBindGroup({
    layout: dataBindGroupLayout,
    bindings: [{
      binding: dataBinding,
      resource: { buffer: dataBuffer, offset: 0, size: arr.byteLength }
    }]
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setBindGroup(bindGroupIndex, dataBindGroup);
  passEncoder.setPipeline(pipeline);
  passEncoder.dispatch(arr.length, 1, 1);
  passEncoder.endPass();

  device.getQueue().submit([commandEncoder.finish()]);

  // get result
  const resultArrayBuffer = await dataBuffer.mapReadAsync();
  log(`GPU square time: ${Math.round(performance.now() - now)} ms`);
  const result = new Uint32Array(resultArrayBuffer);
  console.log(result.length == arr.length);
  console.log(result);
};

const resetData = (arr) => {
  arr.forEach((v, index) => {
    arr[index] = Math.floor(Math.random() * Math.floor(MAX_ORIGINAL_VALUE));
  });
};

const createComputeShader = (length) => {
  // FIXME: Replace with non-MSL.
  return device.createShaderModule({ code: `
    #include <metal_stdlib>

    struct Data {
        device unsigned* numbers [[id(${dataBinding})]];
    };

    kernel void squares_main(device Data& data [[buffer(${bindGroupIndex})]], unsigned gid [[thread_position_in_grid]])
    {
        if (gid >= ${length})
            return;

        unsigned original = data.numbers[gid];
        data.numbers[gid] = original * original;
    }
    ` 
  });
}

const getLength = (index) => {
  return 1 << (index + 10);
};

const log = (str) => {
  logElement.innerText += str + '\n';
};

window.addEventListener('DOMContentLoaded', main);