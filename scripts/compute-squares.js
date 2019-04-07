const MAX_THREAD_NUM = 1024;
const MAX_GROUP_NUM = 2048;
const MAX_ORIGINAL_VALUE = 94906265; // Largest safe square root in JS.

let logElement;
let selectBox;

let device;
let bitonicSortProgram1;
let bitonicSortProgram2;
let bitonicSortProgram2UniformLocation;

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

  // const initializeResult = initializeComputeProgram();
  // if (!initializeResult) {
  //   return;
  // }

  compute();
};

const compute = async () => {
  const length = getLength(selectBox.selectedIndex);
  console.log(`square test: ${length}`);
  const arr = new Float32Array(length);
  resetData(arr);

  await computeCPU(arr.slice(0));
  await computeGPU(arr.slice(0));

  selectBox.disabled = false;
  console.log(`---`);
};

const computeCPU = async (arr) => {
  const now = performance.now();
  arr.foreach((value, index) => {
    arr[index] = value * value;
  });
  log(`CPU square time: ${Math.round(performance.now() - now)} ms`)
  console.log(arr);
};

const computeGPU = async (arr) => {
  const now = performance.now();

  const length = arr.length;

  const threadgroupsPerGrid = Math.max(1, length / MAX_THREAD_NUM);

  // create ShaderStorageBuffer
  // const ssbo = context.createBuffer();
  // context.bindBuffer(context.SHADER_STORAGE_BUFFER, ssbo);
  // context.bufferData(context.SHADER_STORAGE_BUFFER, arr, context.STATIC_DRAW);
  // context.bindBufferBase(context.SHADER_STORAGE_BUFFER, 0, ssbo);

  // // execute ComputeShader
  // context.useProgram(bitonicSortProgram1);
  // context.dispatchCompute(threadgroupsPerGrid, 1, 1);
  // context.memoryBarrier(context.SHADER_STORAGE_BARRIER_BIT);

  // if (threadgroupsPerGrid > 1) {
  //   for (let k = threadgroupsPerGrid; k <= length; k <<= 1) {
  //     for (let j = k >> 1; j > 0; j >>= 1) {
  //       // execute ComputeShader
  //       context.useProgram(bitonicSortProgram2);
  //       context.uniform4uiv(bitonicSortProgram2UniformLocation, new Uint32Array([k, j, 0, 0]));
  //       context.dispatchCompute(threadgroupsPerGrid, 1, 1);
  //       context.memoryBarrier(context.SHADER_STORAGE_BARRIER_BIT);
  //     }
  //   }
  // }

  // // get result
  // const result = new Float32Array(length);
  // context.getBufferSubData(context.SHADER_STORAGE_BUFFER, 0, result);
  log(`GPU sort time: ${Math.round(performance.now() - now)} ms`);
  console.log(result);
};

const resetData = (arr) => {
  arr.foreach((v, index) => {
    arr[index] = Math.floor(Math.random() * Math.floor(MAX_ORIGINAL_VALUE));
  });
};

// const initializeComputeProgram = () => {
//   // ComputeShader source
//   // language=GLSL
//   const computeShaderSource1 = `#version 310 es
//     layout (local_size_x = ${MAX_THREAD_NUM}, local_size_y = 1, local_size_z = 1) in;
//     layout (std430, binding = 0) buffer SSBO {
//       float data[];
//     } ssbo;
//     shared float sharedData[${MAX_THREAD_NUM}];
    
//     void main() {
//       sharedData[gl_LocalInvocationID.x] = ssbo.data[gl_GlobalInvocationID.x];
//       memoryBarrierShared();
//       barrier();
      
//       uint offset = gl_WorkGroupID.x * gl_WorkGroupSize.x;
      
//       float tmp;
//       for (uint k = 2u; k <= gl_WorkGroupSize.x; k <<= 1) {
//         for (uint j = k >> 1; j > 0u; j >>= 1) {
//           uint ixj = (gl_GlobalInvocationID.x ^ j) - offset;
//           if (ixj > gl_LocalInvocationID.x) {
//             if ((gl_GlobalInvocationID.x & k) == 0u) {
//               if (sharedData[gl_LocalInvocationID.x] > sharedData[ixj]) {
//                 tmp = sharedData[gl_LocalInvocationID.x];
//                 sharedData[gl_LocalInvocationID.x] = sharedData[ixj];
//                 sharedData[ixj] = tmp;
//               }
//             }
//             else
//             {
//               if (sharedData[gl_LocalInvocationID.x] < sharedData[ixj]) {
//                 tmp = sharedData[gl_LocalInvocationID.x];
//                 sharedData[gl_LocalInvocationID.x] = sharedData[ixj];
//                 sharedData[ixj] = tmp;
//               }
//             }
//           }
//           memoryBarrierShared();
//           barrier();
//         }
//       }
//       ssbo.data[gl_GlobalInvocationID.x] = sharedData[gl_LocalInvocationID.x];
//     }
//     `;

//   // create WebGLProgram for ComputeShader
//   bitonicSortProgram1 = createComputeProgram(computeShaderSource1);
//   if (!bitonicSortProgram1) {
//     return false;
//   }

//   // language=GLSL
//   const computeShaderSource2 = `#version 310 es
//     layout (local_size_x = ${MAX_THREAD_NUM}, local_size_y = 1, local_size_z = 1) in;
//     layout (std430, binding = 0) buffer SSBO {
//       float data[];
//     } ssbo;
//     uniform uvec4 numElements;
    
//     void main() {
//        float tmp;
//       uint ixj = gl_GlobalInvocationID.x ^ numElements.y;
//       if (ixj > gl_GlobalInvocationID.x)
//       {
//         if ((gl_GlobalInvocationID.x & numElements.x) == 0u)
//         {
//           if (ssbo.data[gl_GlobalInvocationID.x] > ssbo.data[ixj])
//           {
//             tmp = ssbo.data[gl_GlobalInvocationID.x];
//             ssbo.data[gl_GlobalInvocationID.x] = ssbo.data[ixj];
//             ssbo.data[ixj] = tmp;
//           }
//         }
//         else
//         {
//           if (ssbo.data[gl_GlobalInvocationID.x] < ssbo.data[ixj])
//           {
//             tmp = ssbo.data[gl_GlobalInvocationID.x];
//             ssbo.data[gl_GlobalInvocationID.x] = ssbo.data[ixj];
//             ssbo.data[ixj] = tmp;
//           }
//         }
//       }
//     }
//     `;

//   // create WebGLProgram for ComputeShader
//   bitonicSortProgram2 = createComputeProgram(computeShaderSource2);
//   if (!bitonicSortProgram2) {
//     return false;
//   }
//   bitonicSortProgram2UniformLocation = context.getUniformLocation(bitonicSortProgram2, 'numElements');

//   return true;
// };

const getLength = (index) => {
  return 1 << (index + 10);
};

const log = (str) => {
  logElement.innerText += str + '\n';
};


window.addEventListener('DOMContentLoaded', main);