struct MetaUnary{
    input1_layout : MatrixLayout,
    operation : u32,
    scalar1 : u32, //optionally scalar value
    scalar2 : u32,
}

struct MetaBinary{
    input1_layout : MatrixLayout,
    input2_layout : MatrixLayout,
    operation : u32,
}

//(M X N) * (N X K)
struct MetaInfoMatMul{
    b : u32, //batch_count ("normal" matmul = 1)
    m : u32, 
    n : u32, 
    k : u32,
    input1 : MatrixLayout, 
    input2 : MatrixLayout
}

struct MetaInfoReduce{
    input_layout : MatrixLayout,
    operation : u32,
    dimensions : u32, //Dimensions to Reduce Over(if ith bit is set -> Reduce over ith Dimension maximal 5)
}

//Layout Information
struct MatrixLayout{
    shape1 : u32, 
    shape2 : u32, 
    shape3 : u32, 
    shape4 : u32, 
    shape5 : u32, 
    stride1 : u32, 
    stride2 : u32, 
    stride3 : u32, 
    stride4 : u32, 
    stride5 : u32, 
    offset : u32,
    length : u32, //length if continues, else 0 
}


@group(0) @binding(0)
var<storage, read_write> v_dest: array<u32>;

@group(0) @binding(0) 
var<storage, read_write> v_dest_u32: array<u32>; //Output for U8, and U32

@group(0) @binding(1)
var<uniform> op_unary : MetaUnary;

@group(0) @binding(1)
var<uniform> op_binary : MetaBinary;

@group(0) @binding(1)
var<uniform> op_matmul : MetaInfoMatMul;

@group(0) @binding(1)
var<uniform> op_reduce : MetaInfoReduce;

@group(0) @binding(1)
var<uniform> op_input_matrix : array<MatrixLayout, 3>;

@group(0) @binding(2)
var<storage> v_input1: array<u32>;

@group(0) @binding(3)
var<storage> v_input2: array<u32>;

fn set_output_u8(m : u32, v1 : u32, v2 : u32, v3 : u32, v4 : u32){
    let value = ((v1 & 0xFF) << 12) | ((v2 & 0xFF) << 8) | ((v3 & 0xFF) << 4) | (v4 & 0xFF);
    v_dest_u32[m] = value;
}

fn get_output_u8(m : u32) -> array<u32,4>{
    let value = v_dest_u32[m];
    let v4 = value & 0xFF;
    let v3 = (value >> 4) & 0xFF;
    let v2 = (value >> 8) & 0xFF;
    let v1 = (value >> 12) & 0xFF;
    return array(v1,v2,v3,v4);
}



struct MatrixIndex{
    id : u32,
    is_valid : bool
}

fn get_index(l : MatrixLayout, index : u32) -> MatrixIndex{
    if l.length != 0{ //Continues memory:
        if index < l.length{
            return MatrixIndex((l.offset + index), true);
        }
        return MatrixIndex(0, false);
    }
    else { //not continues:
        let length = l.shape1 * l.shape2 * l.shape3 * l.shape4 * l.shape5;
        if index >= l.length{
            return MatrixIndex(0, false);
        }
       
        let s1 = index % (length);
        let s2 = index % (l.shape1 * l.shape2 * l.shape3 * l.shape4);
        let s3 = index % (l.shape1 * l.shape2 * l.shape3);
        let s4 = index % (l.shape1 * l.shape2);
        let s5 = index % (l.shape1);

        let new_index = s1 * l.stride1 + s2 * l.stride2 + s3 * l.stride3 + s4 * l.stride4 + s5 * l.stride5;
         return MatrixIndex(new_index, true);
    }
}

//all Unary Operations(No Input)
fn set_unary(operation : u32, id : u32, x : u32, scalar1 : u32, scalar2 : u32){

    switch operation {
        case 0u{ // set 0
            v_dest[id] = 0u;
        }
        case 1u{ //set 1
            v_dest[id] = 1u;
        }

        //inplace operators
        case 2u{ 
            v_dest[id] += 1u;
        }
        case 3u{ 
            v_dest[id] -= 1u;
        }
        case 4u{  
            v_dest[id] = abs(x);
        }case 38u{   //Relu
            v_dest[id] = max(0u, x) ;
        }
        case 43u{ //Identity
            v_dest[id] = x;
        }
        case 44u{ //square
            v_dest[id] = x * x;
        }
        case 51u{//Affine
            v_dest[id] = x * scalar1 + scalar2;
        }
        case 101u{ //add
            v_dest[id] = x + scalar1;
        }
        case 102u{ //mult
            v_dest[id] = x * scalar1;
        }
        case 103u{ //minus
            v_dest[id] = x - scalar1;
        }
        case 104u{ //div
            v_dest[id] = x / scalar1;
        }
        case 105u{ //max
            v_dest[id] = max(x, scalar1);
        }
        case 106u{ //min
            v_dest[id] = min(x, scalar1);
        }
        default{

        }
    }
}

@compute
@workgroup_size(64,1,1)
fn unary_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let pos1 = get_index(op_unary.input1_layout, id);
    if(pos1.is_valid){
        let x = v_dest[pos1.id];
        set_unary(op_unary.operation,id, x, op_unary.scalar1, op_unary.scalar2);
    }
}


@compute
@workgroup_size(64,1,1)
fn unary_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let pos1 = get_index(op_unary.input1_layout, id);
    if(pos1.is_valid){
        let x = v_input1[pos1.id];
        set_unary(op_unary.operation,id, x, op_unary.scalar1, op_unary.scalar2);
    }
}



fn set_binary(operation : u32, id : u32, x : u32, y : u32){
    switch(operation){
        case 0u{
            v_dest[id] = y;
        }
        case 1u{ //add
            v_dest[id] = x + y;
        }
        case 2u{ //mult
            v_dest[id] = x * y;
        }
        case 3u{ //minus
            v_dest[id] = x - y;
        }
        case 4u{ //div
            v_dest[id] = x / y;
        }
        case 5u{ //max
            v_dest[id] = max(x, y);
        }
        case 6u{ //min
            v_dest[id] = min(x, y);
        }
        default{

        }
    }
}


@compute
@workgroup_size(64,1,1)
fn binary_buffer_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let pos1 = get_index(op_unary.input1_layout, id);
    if(pos1.is_valid){
        set_binary(op_unary.operation, id, v_dest[id], v_input1[pos1.id]);
    }
   
}

@compute
@workgroup_size(64,1,1)
fn binary_buffer_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let pos1 = get_index(op_binary.input1_layout, id);
    let pos2 = get_index(op_binary.input2_layout, id);
    if(pos1.is_valid){
        set_binary(op_binary.operation, id, v_input1[pos1.id], v_input2[pos2.id]);
    }
}



@compute
@workgroup_size(8,8,1)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let b = global_id.z;

    if(x >= op_matmul.k){
        return;
    }
    if(y >= op_matmul.m){
        return;
    }

    let output_size_of_one_batch = op_matmul.m * op_matmul.k; 

    let input1_offset = op_matmul.input1.offset;
    let input2_offset = op_matmul.input2.offset;

    //let input1_stride_m = op_matmul.n;
    //let input1_stride_n = 1u;
    //let input1_stride_b = op_matmul.m * op_matmul.n;
    
    //let input2_stride_n = op_matmul.k;
    //let input2_stride_k = 1u;
    //let input2_stride_b = op_matmul.n * op_matmul.k;

    let input1_stride_b = op_matmul.input1.stride3;
    let input1_stride_m = op_matmul.input1.stride4;
    let input1_stride_n = op_matmul.input1.stride5;

    let input2_stride_b = op_matmul.input2.stride3;
    let input2_stride_n = op_matmul.input2.stride4;
    let input2_stride_k = op_matmul.input2.stride5;

    var sum = 0u;
    for (var i = 0u; i < op_matmul.n; i++){
        sum +=  
        v_input1[b * input1_stride_b  + input1_stride_m * y + i * input1_stride_n + input1_offset] 
      * v_input2[b * input2_stride_b  + input2_stride_n * i + x * input2_stride_k + input2_offset];
    }
    
    v_dest[b * output_size_of_one_batch + y * op_matmul.k + x] = sum;
}


// @compute
// @workgroup_size(1,1,1)
// fn reduce_from_buffer_old() {
//     // let id = global_id.x;
//     // if(id >= op_unary.length){
//     //     return;
//     // }

//     switch(op_reduce.operation){
//         case 0u{ //sum
//             var sum = 0u;
//             for (var i = 0u; i < op_reduce.length; i++){
//                 sum += v_input1[i];
//             }
//             v_dest[0] = sum;
//         }
//         case 1u{ //min
//             var sum = v_input1[0];
//             for (var i = 0u; i < op_reduce.length; i++){
//                 sum = min(sum,v_input1[i]);
//             }
//             v_dest[0] = sum;
//         }
//         case 2u{ //max
//             var sum = v_input1[0];
//             for (var i = 0u; i < op_reduce.length; i++){
//                 sum = max(sum,v_input1[i]);
//             }
//             v_dest[0] = sum;
//         }
//         case 3u{//ArgMin
//             var sum = v_input1[0];
//             var index = 0u;
//             for (var i = 0u; i < op_reduce.length; i++){
//                 if v_input1[i] < sum{
//                     sum = v_input1[i];
//                     index = i;
//                 }
//             }
//             v_dest[0] = index;
//         }
//         case 4u{//ArgMax
//             var sum = v_input1[0];
//             var index = 0u;
//             for (var i = 0u; i < op_reduce.length; i++){
//                 if v_input1[i] > sum{
//                     sum = v_input1[i];
//                     index = i;
//                 }
//             }
//             v_dest[0] = index;
//         }
//         default{

//         }
//     }
// }



fn linear_index(indices: array<u32, 5>, strides: array<u32, 5>) -> u32 {
     return indices[0] * strides[0] +
            indices[1] * strides[1] +
            indices[2] * strides[2] +
            indices[3] * strides[3] +
            indices[4] * strides[4];
}

   
var<workgroup> sharedSums: array<u32, 64>;  
@compute @workgroup_size(64)
fn reduce_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Compute the shape of the output buffer
    var shape = array<u32, 5>(op_reduce.input_layout.shape1, op_reduce.input_layout.shape2, op_reduce.input_layout.shape3, op_reduce.input_layout.shape4, op_reduce.input_layout.shape5);
    var strides = array<u32, 5>(op_reduce.input_layout.stride1, op_reduce.input_layout.stride2, op_reduce.input_layout.stride3, op_reduce.input_layout.stride4, op_reduce.input_layout.stride5);
    var out_shape: array<u32, 5>;
    for (var i = 0u; i < 5u; i = i + 1u) {
        if ((op_reduce.dimensions & (1u << i)) != 0u) {
            out_shape[i] = 1u;
        } else {
            out_shape[i] = shape[i];
        }
    }

    // Compute the indices in the input and output buffers
    var indices: array<u32, 5>;
    var out_indices: array<u32, 5>;
    var rem = index;
    for (var i = 0u; i < 5u; i = i + 1u) {
        indices[i] = rem % shape[i];
        rem = rem / shape[i];
        if (out_shape[i] != 1u) {
            out_indices[i] = indices[i];
        } else {
            out_indices[i] = 0u;
        }
    }

    // Load the input data into shared memory
    let linear_in_index = op_reduce.input_layout.offset + linear_index(indices, strides);
    if (index < arrayLength(&v_input1)) {
        sharedSums[index % 64] = v_input1[linear_in_index];
    } else {
        sharedSums[index % 64] = 0u;
    }

    workgroupBarrier();

    // Perform reduction within the workgroup
    for (var offset = 32u; offset > 0u; offset = offset >> 1u) {
        if (index % 64 < offset) {
            sharedSums[index % 64] = sharedSums[index % 64] + sharedSums[index % 64 + offset];
        }
        workgroupBarrier();
    }

    // Write the result to the output buffer
    if (index % 64 == 0u) {
        let linear_out_index = linear_index(out_indices, strides);
        v_dest[linear_out_index] = sharedSums[0];
    }
}

fn bool_to_int(b : bool) -> u32{
    if b{
        return 1u;
    }
    return 0u;
}

@compute
@workgroup_size(64,1,1)
fn cmp_buffer_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let pos1 = get_index(op_binary.input1_layout, id);
    let pos2 = get_index(op_binary.input2_layout, id);
    if(!pos1.is_valid){
        return;
    }
    
    let x = v_input1[pos1.id];
    let y = v_input2[pos2.id];

    switch(op_binary.operation){
        case 0u: { //eq
            v_dest_u32[id] = bool_to_int(x == y);
        }
        case 1u: {//ne
            v_dest_u32[id] = bool_to_int(x != y);
        }
        case 2u: {//lt
            v_dest_u32[id] = bool_to_int(x < y);
        }
        case 3u: {//LE
            v_dest_u32[id] = bool_to_int(x <= y);
        }
        case 4u: {//GT
            v_dest_u32[id] = bool_to_int(x > y);
        }
        case 5u: {//GE
            v_dest_u32[id] = bool_to_int(x >= y);
        }
        default:{
            
        }
    }
}