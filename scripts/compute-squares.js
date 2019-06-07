// This demo is based on webgl-compute-bitonicSort, found at https://github.com/9ballsyndrome/WebGL_Compute_shader.

const MAX_THREAD_NUM = 1024;
const MAX_GROUP_NUM = 2048;
const MAX_ORIGINAL_VALUE = 64; // 94906265; // Largest safe square root in JS.

let logElement;
let selectBox;

let device;
const dataBinding = 0;
const uniformsBinding = 1;
const bindGroupIndex = 0;
const uniformsGroupIndex = 1;


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

  if (!navigator.gpu) {
    document.body.className = 'error';
    return;
  }

  // Create GPUDevice
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();

  compute();
};

const compute = async () => {
  const length = getLength(selectBox.selectedIndex);
  console.log(`sort test: ${length}`);
  const arr = new Uint32Array(length);
  resetData(arr);

  await computeCPU(arr.slice(0));
  await computeGPU(arr.slice(0));

  selectBox.disabled = false;
  console.log(`---`);
};

const computeCPU = async (arr) => {
  const now = performance.now();
  arr.sort((a, b) => { return a - b; });
  log(`CPU sort time: ${Math.round(performance.now() - now)} ms`);
  console.log(`CPU sort result validation: ${validateSorted(arr) ? 'success' : 'failure'}`);
  console.log(arr);
};

function createBufferWithData(device, descriptor, data, offset = 0) {
  const mappedBuffer = device.createBufferMapped(descriptor);
  const dataArray = new Uint8Array(mappedBuffer[1]);
  dataArray.set(new Uint8Array(data), offset);
  mappedBuffer[0].unmap();
  return mappedBuffer[0];
}

async function mapWriteDataToBuffer(buffer, data, offset = 0) {
  const arrayBuffer = await buffer.mapWriteAsync();
  const writeArray = new Uint8Array(arrayBuffer);
  writeArray.set(new Uint8Array(data), offset);
  buffer.unmap();
}

const computeGPU = async (arr) => {
  const now = performance.now();

  const shaderModule = createComputeShader(arr.length);
  const pipeline0 = device.createComputePipeline({ 
    computeStage: { module: shaderModule, entryPoint: "sort_0" } 
  });
  
  const dataBuffer = createBufferWithData(device, { 
    size: arr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.MAP_READ
  }, arr.buffer);

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

  const pipeline1 = device.createComputePipeline({
    computeStage: { module: shaderModule, entryPoint: "sort_1" }
  });

  const uniformsBufferSize = 2 * Uint32Array.BYTES_PER_ELEMENT;
  const uniformsBuffer = device.createBuffer({
    size: uniformsBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.MAP_WRITE
  });

  const uniformsBindGroupLayout = device.createBindGroupLayout({
    bindings: [{ binding: uniformsBinding, visibility: GPUShaderStageBit.COMPUTE, type: "storage-buffer" }]
  });

  const uniformsBindGroup = device.createBindGroup({
    layout: uniformsBindGroupLayout,
    bindings: [{
      binding: uniformsBinding,
      resource: { buffer: uniformsBuffer, offset: 0, size: uniformsBufferSize }
    }]
  });

  const commandEncoder = device.createCommandEncoder();
  let passEncoder = commandEncoder.beginComputePass();
  passEncoder.setBindGroup(bindGroupIndex, dataBindGroup);
  passEncoder.setPipeline(pipeline0);

  const threadgroupsPerGrid = Math.max(1, arr.length / MAX_THREAD_NUM);
  passEncoder.dispatch(threadgroupsPerGrid, 1, 1);
  passEncoder.endPass();

  if (threadgroupsPerGrid > 1) {
    let numElementsArray = new Uint32Array(2);

    for (let k = threadgroupsPerGrid; k <= arr.length; k <<= 1) {
      for (let j = k >> 1; j > 0; j >>= 1) {
        numElementsArray[0] = k;
        numElementsArray[1] = j;
        await mapWriteDataToBuffer(uniformsBuffer, numElementsArray.buffer);

        passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(bindGroupIndex, dataBindGroup);
        passEncoder.setBindGroup(uniformsGroupIndex, uniformsBindGroup);
        passEncoder.setPipeline(pipeline1);
        passEncoder.dispatch(threadgroupsPerGrid, 1, 1);
        passEncoder.endPass();
      }
    }
  }

  device.getQueue().submit([commandEncoder.finish()]);

  // get result
  const resultArrayBuffer = await dataBuffer.mapReadAsync();
  log(`GPU sort time: ${Math.round(performance.now() - now)} ms`);
  const result = new Uint32Array(resultArrayBuffer);
  console.log(`GPU sort result validation: ${validateSorted(result) ? 'success' : 'failure'}`);
  console.log(result);
};

const resetData = (arr) => {
  arr.forEach((v, index) => {
    arr[index] = Math.floor(Math.random() * Math.floor(MAX_ORIGINAL_VALUE));
  });
};

const validateSorted = (arr) => {
  const length = arr.length;
  for (let i = 0; i < length; i++) {
    if (i !== length - 1 && arr[i] > arr[i + 1]) {
      console.log('validation error:', i, arr[i], arr[i + 1]);
      return false;
    }
  }
  return true;
};

const createComputeShader = (length) => {
  // FIXME: Replace with non-MSL.
  return device.createShaderModule({ code: `
  #include <metal_stdlib>

  using namespace metal;

  struct Data {
      device unsigned* numbers [[id(${dataBinding})]];
  };
  
  kernel
  void sort_0(device Data& data         [[buffer(${bindGroupIndex})]],
              unsigned globalID         [[thread_position_in_grid]],
              unsigned localID          [[thread_position_in_threadgroup]],
              unsigned threadgroupID    [[threadgroup_position_in_grid]],
              unsigned threadgroupSize  [[threads_per_threadgroup]])
  {
      threadgroup unsigned sharedData[${MAX_THREAD_NUM}];

      sharedData[localID] = data.numbers[globalID];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup_barrier(mem_flags::mem_none);
      
      unsigned offset = threadgroupID * threadgroupSize;
      
      unsigned temp;
      for (unsigned k = 2; k <= threadgroupSize; k <<= 1) {
          for (unsigned j = k >> 1; j > 0; j >>= 1) {
              unsigned ixj = (globalID ^ j) - offset;
              if (ixj > localID) {
                  if ((globalID & k) == 0) {
                      if (sharedData[localID] > sharedData[ixj]) {
                          temp = sharedData[localID];
                          sharedData[localID] = sharedData[ixj];
                          sharedData[ixj] = temp;
                      }
                  } else {
                      if (sharedData[localID] < sharedData[ixj]) {
                          temp = sharedData[localID];
                          sharedData[localID] = sharedData[ixj];
                          sharedData[ixj] = temp;
                      }
                  }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              threadgroup_barrier(mem_flags::mem_none);
          }
      }
      data.numbers[globalID] = sharedData[localID];
  }
  
  struct Uniform {
      device unsigned* numElements [[id(${uniformsBinding})]];
  };
  
  kernel
  void sort_1(device Data& data           [[buffer(${bindGroupIndex})]],
              device Uniform& uniforms    [[buffer(${uniformsGroupIndex})]],
              unsigned globalID           [[thread_position_in_grid]])
  {
      unsigned temp;
      unsigned ixj = globalID ^ uniforms.numElements[1];
      if (ixj > globalID) {
          if ((globalID & uniforms.numElements[0]) == 0) {
              if (data.numbers[globalID] > data.numbers[ixj]) {
                  temp = data.numbers[globalID] = data.numbers[ixj];
                  data.numbers[globalID] = data.numbers[ixj];
                  data.numbers[ixj] = temp;
              }
          } else {
              if (data.numbers[globalID] < data.numbers[ixj]) {
                  temp = data.numbers[globalID] = data.numbers[ixj];
                  data.numbers[globalID] = data.numbers[ixj];
                  data.numbers[ixj] = temp;
              }
          }
      }
  }`
  });
}

const getLength = (index) => {
  //return 1 << (index + 10);
  return 128;
};

const log = (str) => {
  logElement.innerText += str + '\n';
};

window.addEventListener('DOMContentLoaded', main);