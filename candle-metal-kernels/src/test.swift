
import Metal
import MetalPerformanceShadersGraph



let type = MTLDataType.float;
let dataType = type;
var B = 2;
var M = 2;
var N = 4;
var K = 3;
var A_trans = false;
var B_trans = false;
var D_trans = false;
var alpha = Float(1.0);
var beta = Float(0.0);
var batched = B > 1;
var fused_activation = false;
var fused_bias = false;
let constants = MTLFunctionConstantValues()
constants.setConstantValue(&M, type: .uint, index: 0)
constants.setConstantValue(&N, type: .uint, index: 1)
constants.setConstantValue(&K, type: .uint, index: 2)
constants.setConstantValue(&A_trans, type: .bool, index: 10)
constants.setConstantValue(&B_trans, type: .bool, index: 11)
constants.setConstantValue(&D_trans, type: .bool, index: 13)
constants.setConstantValue(&alpha, type: .float, index: 20)
constants.setConstantValue(&beta, type: .float, index: 21)
constants.setConstantValue(&batched, type: .bool, index: 100)
constants.setConstantValue(&fused_activation, type: .bool, index: 101)
constants.setConstantValue(&fused_bias, type: .bool, index: 50001)


var M_simd = UInt16(16)
var N_simd = UInt16(16)
var K_simd = UInt16(32)
var M_splits = UInt16(2)
var N_splits = UInt16(2)
constants.setConstantValue(&M_simd, type: .ushort, index: 200)
constants.setConstantValue(&N_simd, type: .ushort, index: 201)
constants.setConstantValue(&K_simd, type: .ushort, index: 202)
constants.setConstantValue(&M_splits, type: .ushort, index: 210)
constants.setConstantValue(&N_splits, type: .ushort, index: 211)

let M_group = M_simd * M_splits
let N_group = N_simd * N_splits

// Satisfy Metal API validation.
#if DEBUG
do {
  var garbage: SIMD4<UInt64> = .zero
  constants.setConstantValue(&garbage, type: .bool, index: 102)
  constants.setConstantValue(&garbage, type: .bool, index: 103)
  constants.setConstantValue(&garbage, type: .bool, index: 113)
  constants.setConstantValue(&garbage, type: .bool, index: 50000)
}
#endif
print(constants)

let device = MTLCopyAllDevices().first!
device.shouldMaximizeConcurrentCompilation = true

var libraryURL = URL.init(string: "/Users/nicolas/src/candle/candle-metal-kernels/")!;
libraryURL.append(component: "src")
libraryURL.append(component: "libMetalFlashAttention.metallib")
let library = try! device.makeLibrary(URL: libraryURL)

var name: String
    switch dataType {
    case .half: name = "hgemm"
    case .float: name = "sgemm"
    default: fatalError()
    }
let function = try! library.makeFunction(
  name: name, constantValues: constants)

let A_block_length = M_group * K_simd
let B_block_length = K_simd * N_group

var blockElements = A_block_length + B_block_length;
if (M % 8 != 0) && (N % 8 != 0) {
  let C_block_length = M_group * N_group;
  blockElements = max(C_block_length, blockElements)
}
if fused_bias {
  if D_trans {
    blockElements = max(blockElements, M_group)
  } else {
    blockElements = max(blockElements, N_group)
  }
}
// let blockBytes = blockElements * UInt16(dataType.size)
let elementSize = 4
let blockBytes = blockElements * UInt16(elementSize)

func ceilDivide(target: Int, granularity: UInt16) -> Int {
  (target + Int(granularity) - 1) / Int(granularity)
}
var gridSize = MTLSize(
  width: ceilDivide(target: N, granularity: N_group),
  height: ceilDivide(target: M, granularity: M_group),
  depth: 1)
let groupSize = MTLSize(
  width: Int(32 * M_splits * N_splits),
  height: 1,
  depth: 1)

let commandQueue = device.makeCommandQueue()!
let commandBuffer = commandQueue.makeCommandBuffer()!
let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: MTLDispatchType.serial)!
let pipeline = try device.makeComputePipelineState(function: function)

let threadgroupMemoryLength = blockBytes;
print(threadgroupMemoryLength)
encoder.setComputePipelineState(pipeline)
encoder.setThreadgroupMemoryLength(Int(threadgroupMemoryLength), index: 0)


let rowsA = M;
let columnsA = K;
let rowsB = K;
let columnsB = N;
let rowsC = M;
let columnsC = N;
var arrayA = [Float](repeating: 0, count: B * rowsA * columnsA)

var arrayB = [Float](repeating: 0, count: B * rowsB * columnsB)

var arrayC = [Float](repeating: 0, count: B * rowsC * columnsC)
for i in 0..<arrayA.count {
  arrayA[i] = Float(i)
}

for i in 0..<arrayB.count {
  arrayB[i] = Float(i)
}

let bufferA = device.makeBuffer(bytes: arrayA, length: B * rowsA * columnsA * MemoryLayout<Float>.stride, options: [])

let bufferB = device.makeBuffer(bytes: arrayB, length: B * rowsB * columnsB * MemoryLayout<Float>.stride, options: [])

let bufferC = device.makeBuffer(length: B * rowsC * columnsC * MemoryLayout<Float>.stride, options: [])

print(arrayA)
print(arrayB)


encoder.setBuffer(bufferA, offset: 0, index: 0)
encoder.setBuffer(bufferB, offset: 0, index: 1)
encoder.setBuffer(bufferC, offset: 0, index: 2)
var gridZ: Int = B
if batched{
  func byteStride(shape: [Int]) -> Int {
    let rank = shape.count
    var output = elementSize * shape[rank - 2] * shape[rank - 1]
    if shape.dropLast(2).reduce(1, *) == 1 {
      output = 0
    }
    return output
  }
  let byteStrideA = M*K*elementSize
  let byteStrideB = N*K*elementSize
  let byteStrideC = M*N*elementSize
  
  let byteStrideD = 0
  // if let shapeD = tensors.d?.shape {
  //   let rank = shapeD.count
  //   byteStrideD = elementSize * shapeD[rank - 1]
  //   if shapeD.dropLast(1).reduce(1, *) == 1 {
  //     byteStrideD = 0
  //   }
  // }
  withUnsafeTemporaryAllocation(
    of: SIMD4<UInt64>.self, capacity: gridZ
  ) { buffer in
    for i in 0..<buffer.count {
      buffer[i] = SIMD4(
        UInt64(truncatingIfNeeded: i * byteStrideA),
        UInt64(truncatingIfNeeded: i * byteStrideB),
        UInt64(truncatingIfNeeded: i * byteStrideC),
        UInt64(truncatingIfNeeded: i * byteStrideD))
    }
    
    let bufferLength = buffer.count * MemoryLayout<SIMD4<UInt64>>.stride
    assert(MemoryLayout<SIMD4<UInt64>>.stride == 8 * 4)
    encoder.setBytes(buffer.baseAddress!, length: bufferLength, index: 10)
    print("BATCHED")
    print(buffer)
  }
}
gridSize.depth = gridZ


print(gridSize, groupSize)
encoder.dispatchThreadgroups(
  gridSize, threadsPerThreadgroup: groupSize
)
encoder.endEncoding()
commandBuffer.commit()

commandBuffer.waitUntilCompleted()
      var contents = bufferC!.contents();

      var count = B * rowsA * columnsB;

    var typedPointer = contents.bindMemory(to: Float.self, capacity: count)

    var bufferedPointer = UnsafeBufferPointer(start: typedPointer, count: count)

    print(Array(bufferedPointer))
