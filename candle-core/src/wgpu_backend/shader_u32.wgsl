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
    workgroup_count : u32,
    workgroup_size : u32,
    length : u32, //Length of Reduction(e.g count of elements to sum per output),

    output_to_start_stride1 : u32, //Stride between each new Output Index
    
    output_to_start_shape_stride2 : u32, //After x Outputs use Stride 2 
    output_to_start_stride2 : u32,

    stride_reduction : u32, //The Stride to use for elements in Reduction
}

struct MetaConv2d{
    b : u32, //batch_count ("normal" matmul = 1)
    c_in : u32, //Output Channel, we are using workgroups for all c_out, x, y pairs
    kernel_x : u32,
    kernel_y : u32,
    size_in_x: u32,
    size_in_y : u32,
    stride_batch_out : u32,
    stride_c_out: u32,
    stride_y_out: u32,
    size_y_out : u32,

    stride_batch_input : u32,
    stride_c_in : u32,
    stride_y_in : u32,
    padding : u32,
    stride_conv : u32,
    dialation_conv : u32,
    offset_input : u32,
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
var<storage, read_write> v_dest_f32: array<f32>;

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
var<uniform> op_conv2d : MetaConv2d;


@group(0) @binding(1)
var<uniform> op_input_matrix : array<MatrixLayout, 3>;


@group(0) @binding(2)
var<storage> v_input1: array<u32>;

@group(0) @binding(3)
var<storage> v_input2: array<u32>;

var<workgroup> sharedSums: array<u32, 64>;  //for reduction
var<workgroup> sharedIndex: array<u32, 64>; 

// fn set_output_u8(m : u32, v1 : u32, v2 : u32, v3 : u32, v4 : u32){
//     let value = ((v1 & 0xFF) << 12) | ((v2 & 0xFF) << 8) | ((v3 & 0xFF) << 4) | (v4 & 0xFF);
//     v_dest_u32[m] = value;
// }

// fn get_output_u8(m : u32) -> array<u32,4>{
//     let value = v_dest_u32[m];
//     let v4 = value & 0xFF;
//     let v3 = (value >> 4) & 0xFF;
//     let v2 = (value >> 8) & 0xFF;
//     let v1 = (value >> 12) & 0xFF;
//     return array(v1,v2,v3,v4);
// }


struct MatrixIndex{
    id : u32,
    is_valid : bool
}

const ZERO : u32 = 0;
const ONE : u32 = 1;

@compute
@workgroup_size(64,1,1)
fn convert_to_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let pos1 = get_index(op_unary.input1_layout, id);
    if(pos1.is_valid){
        v_dest_f32[id] = f32(v_input1[pos1.id]);
    }
}


fn get_index(l : MatrixLayout, index : u32) -> MatrixIndex{
    if l.length != 0{ //Continues memory:
        if index < l.length{
            return MatrixIndex((l.offset + index), true);
        }
        return MatrixIndex(0, false);
    }
    else { //not continues:
        let shapes1 = l.shape5;
        let shapes2 =  (shapes1 * l.shape4);
        let shapes3 =  (shapes2 * l.shape3);
        let shapes4 =  (shapes3 * l.shape2);
        let shapes5 =  (shapes4 * l.shape1);
       
        let s1 = (index / shapes4);
        let s2 = (index / shapes3) % (shapes4 / shapes3);
        let s3 = (index / shapes2) % (shapes3 / shapes2);
        let s4 = (index / shapes1) % (shapes2 / shapes1);
        let s5 = index             % (shapes1);

        let new_index = l.offset + s1 * l.stride1 + s2 * l.stride2 + s3 * l.stride3 + s4 * l.stride4 + s5 * l.stride5;
         return MatrixIndex(new_index, true);
    }
}




// fn rand_uniform(value: u32) -> u32 {
//     // Use XORShift algorithm to generate a pseudo-random float
//     // Parameters for XORShift algorithm (adjust as needed)
//     var state: u32 = value ^ 0x5F3759DF; // Initial state, can be any non-zero value
//     state ^= state << 13;
//     state ^= state >> 17;
//     state ^= state << 5;

//     // Convert u32 to a float between 0 and 1
//     // Divide by maximum u32 value to get a float in [0, 1)
//     return state;
// }

//all Unary Operations(No Input)
fn set_unary(operation : u32, id : u32, x : u32, scalar1 : u32, scalar2 : u32){
    switch operation {
        case 0u{ // set 0
            v_dest[id] = ZERO;
        }
        case 1u{ //set 1
            v_dest[id] = ONE;
        }
        case 2u{ 
            v_dest[id] += ONE;
        }
        case 3u{ 
            v_dest[id] -= ONE;
        }
        case 4u{ //Identity
            v_dest[id] = x;
        }
        case 5u{ //square
            v_dest[id] = x * x;
        }
        case 6u{//Affine
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

    var sum = ZERO;
    for (var i = 0u; i < op_matmul.n; i++){
        sum +=  
        v_input1[b * input1_stride_b  + input1_stride_m * y + i * input1_stride_n + input1_offset] 
      * v_input2[b * input2_stride_b  + input2_stride_n * i + x * input2_stride_k + input2_offset];
    }
    
    v_dest[b * output_size_of_one_batch + y * op_matmul.k + x] = sum;
}


@compute
@workgroup_size(64,1,1)
fn reduce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_id = global_id.x;
    let output_index = global_id.y;
    
    //Start Index of the Elements to Reduce

    var start_index = 0u;// = output_index * op_reduce.output_to_start_stride1 + (output_index / op_reduce.output_to_start_shape_stride2) * op_reduce.output_to_start_stride2;
    if op_reduce.output_to_start_shape_stride2 <= 1
    {
        start_index = output_index * op_reduce.output_to_start_stride1;
    }
    else{
        start_index = output_index * op_reduce.output_to_start_stride1 + (output_index / op_reduce.output_to_start_shape_stride2) * op_reduce.output_to_start_stride2;
    }
    
    let length = op_reduce.length; //length of the elements to reduce

    //We split the Reduction into 64 threads -> find the sub region we need to reduce over 
    let start = workgroup_id * op_reduce.workgroup_size;
    let end = min(length, (workgroup_id + 1) * op_reduce.workgroup_size);

    //Now Reduce from start to end
    switch(op_reduce.operation){
        case 0u{ //sum
            var sum = ZERO;
            for (var i = start; i < end; i++){
                let index = get_index(op_reduce.input_layout, start_index + i * op_reduce.stride_reduction).id;
                sum += v_input1[index];
            }
            sharedSums[workgroup_id] = sum;
        }
        case 1u{ //min
            var sum = v_input1[get_index(op_reduce.input_layout, start_index).id];
            for (var i = start + 1; i < end; i++){
                let index = get_index(op_reduce.input_layout, start_index + i * op_reduce.stride_reduction).id;
                sum = min(sum, v_input1[index]);
            }
            sharedSums[workgroup_id] = sum;
        }
        case 2u{ //max
            var sum = v_input1[get_index(op_reduce.input_layout, start_index).id];
            for (var i = start + 1; i < end; i++){
                let index = get_index(op_reduce.input_layout, start_index + i * op_reduce.stride_reduction).id;
                sum = max(sum, v_input1[index]);
            }
            sharedSums[workgroup_id] = sum;
        }
        default{

        }
    }
    
    workgroupBarrier();

    if (workgroup_id == 0){
        let cnt = op_reduce.workgroup_count;
        //Finnaly Sum of all worker threads:

        switch(op_reduce.operation){
            case 0u{ //sum
                var sum = ZERO;
                for (var i = 0u; i < cnt; i++){
                    sum +=  sharedSums[i];
                }
                v_dest[output_index] = sum;
            }
            case 1u{ //min
                var sum = sharedSums[0];
                for (var i = 0u; i < cnt; i++){
                    sum = min(sum, sharedSums[i]);
                }
                v_dest[output_index] = sum;
            }
            case 2u{ //max
                var sum = sharedSums[0];
                for (var i = 0u; i < cnt; i++){
                    sum = max(sum, sharedSums[i]);
                }
                v_dest[output_index] = sum;
            }
            default{

            }
        }
    }
}

@compute
@workgroup_size(64,1,1)
fn reduce_index(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_id = global_id.x;
    let output_index = global_id.y;
    
    //Start Index of the Elements to Reduce

    var start_index = 0u;// = output_index * op_reduce.output_to_start_stride1 + (output_index / op_reduce.output_to_start_shape_stride2) * op_reduce.output_to_start_stride2;
    if op_reduce.output_to_start_shape_stride2 <= 1
    {
        start_index = output_index * op_reduce.output_to_start_stride1;
    }
    else{
        start_index = output_index * op_reduce.output_to_start_stride1 + (output_index / op_reduce.output_to_start_shape_stride2) * op_reduce.output_to_start_stride2;
    }
    
    let length = op_reduce.length; //length of the elements to reduce

    //We split the Reduction into 64 threads -> find the sub region we need to reduce over 
    let start = workgroup_id * op_reduce.workgroup_size;
    let end = min(length, (workgroup_id + 1) * op_reduce.workgroup_size);

    //Now Reduce from start to end
    switch(op_reduce.operation){
        case 3u{//ArgMin
            var sum = v_input1[get_index(op_reduce.input_layout, start_index).id];
            var arg_index = 0u;
            for (var i = start + 1; i < end; i++){
                let index = get_index(op_reduce.input_layout, start_index + i * op_reduce.stride_reduction).id;
                 if v_input1[index] < sum{
                    sum = v_input1[index];
                    arg_index = i;
                }
            }
            sharedSums[workgroup_id] = sum;    
            sharedIndex[workgroup_id] = arg_index;
        }
        case 4u{//ArgMax
            var sum = v_input1[get_index(op_reduce.input_layout, start_index).id];
            var arg_index = 0u;
            for (var i = start + 1; i < end; i++){
                let index = get_index(op_reduce.input_layout, start_index + i * op_reduce.stride_reduction).id;
                 if v_input1[index] > sum{
                    sum = v_input1[index];
                    arg_index = i;
                }
            }
            sharedSums[workgroup_id] = sum;    
            sharedIndex[workgroup_id] = arg_index;
        }
        default{

        }
    }
    
    workgroupBarrier();

    if (workgroup_id == 0){
        let cnt = op_reduce.workgroup_count;
        //Finnaly Sum of all worker threads:
        

        switch(op_reduce.operation){
            case 3u{//ArgMin
                var sum = sharedSums[0];
                var index = 0u;
                for (var i = 0u; i < cnt; i++){
                    if sharedSums[i] < sum{
                        sum = sharedSums[i];
                        index = i;
                    }
                }
                v_dest_u32[output_index] = sharedIndex[index];
            }
            case 4u{//ArgMax
                var sum = sharedSums[0];
                var index = 0u;
                for (var i = 0u; i < cnt; i++){
                    if sharedSums[i] > sum{
                        sum = sharedSums[i];
                        index = i;
                    }
                }
                v_dest_u32[output_index] = sharedIndex[index];
            }
            default{

            }
        }
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


//(N, C_IN, H, W) CONV (C_IN, K_H, K_W) = (N,C_OUT, H_OUT, W_OUT)
//bzgl CONV: Padding x, Padding y, Stride x, stride y, dilation, groups?
@compute
@workgroup_size(8,8,1)
fn conv2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i_out_x = global_id.x;
    let i_out_y = global_id.y;
    let i_c_out = global_id.z;

    let size_y_out = op_conv2d.size_y_out;
    let size_x_out = op_conv2d.stride_y_out;

    if i_out_x >= size_x_out || i_out_y >= size_y_out {
        return;
    }

    let kernel_size_x = op_conv2d.kernel_x;
    let kernel_size_y = op_conv2d.kernel_y;
    let kernel_c_stride = kernel_size_x * kernel_size_y;
    let kernel_y_stride = kernel_size_x;
    let kernel_b_stride = kernel_size_x * kernel_size_y * op_conv2d.c_in;

    //TODO: Pass this Valu above
    let size_in_x = op_conv2d.size_in_x;
    let size_in_y =  op_conv2d.size_in_y;
    let stride_batch_out =  op_conv2d.stride_batch_out;
    let stride_c_out =  op_conv2d.stride_c_out;
    let stride_y_out =  op_conv2d.stride_y_out;

    let stride_batch_input = op_conv2d.stride_batch_input;
    let stride_c_in =  op_conv2d.stride_c_in;
    let stride_y_in =  op_conv2d.stride_y_in;
    let padding =  op_conv2d.padding;
    let stride_conv = op_conv2d.stride_conv;
    let dialation_conv = op_conv2d.dialation_conv;

    //Calculate the top Left Index of the x/y coord 

    let x_coord_offset = i_out_x * stride_conv - padding; //TODO: CALCULATE WITH I32, we need negative numbers for x_coord_offset
    let y_coord_offset = i_out_y * stride_conv - padding;
  

    for (var i_b = 0u; i_b < op_conv2d.b; i_b = i_b + 1u) { //For each Batch:
        var sum = ZERO;
        for (var i_c_in = 0u; i_c_in < op_conv2d.c_in; i_c_in = i_c_in + 1u) { //For each Input Channel:
            let image_offset = i_b * stride_batch_input + i_c_in * stride_c_in ;
            for (var x_k = 0u; x_k < kernel_size_x; x_k = x_k + 1u) { //For each Kernel X
                for (var y_k = 0u; y_k < kernel_size_y; y_k = y_k + 1u) { //For each Kernel X
                    let x_coord = x_coord_offset + dialation_conv * x_k;
                    let y_coord = y_coord_offset + dialation_conv * y_k;
                    if !(x_coord < 0 || y_coord < 0 || x_coord >= size_in_x || y_coord >= size_in_y){ //Ansonsten wäre dieser Index wegen Padding == null 
                        let input_pixel = v_input1[image_offset +  y_coord * stride_y_in + x_coord + op_conv2d.offset_input];
                        sum += v_input2[i_c_out * kernel_b_stride + i_c_in * kernel_c_stride + y_k * kernel_y_stride + x_k] * input_pixel;
                    }
                } 
            }
        }
        v_dest[i_b * stride_batch_out + i_c_out * stride_c_out + stride_y_out * i_out_y + i_out_x] = sum;
    }
}


//(N, C_IN, H, W) CONV (C_IN, K_H, K_W) = (N,C_OUT, H_OUT, W_OUT)
//bzgl CONV: Padding x, Padding y, Stride x, stride y, dilation, groups?
@compute
@workgroup_size(8,8,1)
fn conv2d_transpose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i_out_x = global_id.x + op_conv2d.padding;
    let i_out_y = global_id.y + op_conv2d.padding; //We go through each output 
    let i_c_out = global_id.z; 

    let size_y_out = op_conv2d.size_y_out;
    let size_x_out = op_conv2d.stride_y_out;

    if  global_id.x  >= size_x_out ||  global_id.y >= size_y_out {
        return;
    }

    let kernel_size_x = op_conv2d.kernel_x;
    let kernel_size_y = op_conv2d.kernel_y;
    let kernel_c_stride = kernel_size_x * kernel_size_y;
    let kernel_y_stride = kernel_size_x;
    let kernel_b_stride = kernel_size_x * kernel_size_y * op_conv2d.c_in;

    
    let stride_batch_out =  op_conv2d.stride_batch_out;
    let stride_c_out =  op_conv2d.stride_c_out;
    let stride_y_out =  op_conv2d.stride_y_out;

    let stride_batch_input = op_conv2d.stride_batch_input;
    let stride_c_in =  op_conv2d.stride_c_in;
    let stride_y_in =  op_conv2d.stride_y_in;
    let padding_x = (kernel_size_x - 1);
    let padding_y = (kernel_size_y - 1);
    let stride_conv = op_conv2d.stride_conv;
    let dialation_conv = op_conv2d.dialation_conv;
    let input_dialation = stride_conv;
    let size_in_x = op_conv2d.size_in_x;
    let size_in_y =  op_conv2d.size_in_y;

    //Calculate the top Left Index of the x/y coord 

    let x_coord_offset = i32(i_out_x) - i32(padding_x); //TODO: CALCULATE WITH I32, we need negative numbers for x_coord_offset
    let y_coord_offset = i32(i_out_y) - i32(padding_y);
  

    for (var i_b = 0u; i_b < op_conv2d.b; i_b = i_b + 1u) { //For each Batch:
        var sum = ZERO;
        for (var i_c_in = 0u; i_c_in < op_conv2d.c_in; i_c_in = i_c_in + 1u) { //For each Input Channel:
            let image_offset = i_b * stride_batch_input + i_c_in * stride_c_in ;
            for (var x_k = 0u; x_k < kernel_size_x; x_k = x_k + 1u) { //For each Kernel X
                for (var y_k = 0u; y_k < kernel_size_y; y_k = y_k + 1u) { //For each Kernel X
                    let x_input_offset = dialation_conv * x_k;
                    let y_input_offset = dialation_conv * y_k;

                    let x_coord2 = x_coord_offset + i32(dialation_conv * x_k);
                    let y_coord2 = y_coord_offset + i32(dialation_conv * y_k);
                    if (x_coord2 < 0 || y_coord2 < 0){ //Ansonsten wäre dieser Index wegen Padding == null 
                        continue;
                    }

                    if (u32(x_coord2) % input_dialation) != 0 || ((u32(y_coord2) % input_dialation) != 0){
                        continue;
                    }

                    let x_coord = u32(x_coord2) / input_dialation;
                    let y_coord = u32(y_coord2) / input_dialation;

                    if !(x_coord >= size_in_x || y_coord >= size_in_y){ //Ansonsten wäre dieser Index wegen Padding == null 
                        
                        let input_pixel = v_input1[image_offset +  y_coord * stride_y_in + x_coord + op_conv2d.offset_input];
                        sum += v_input2[i_c_out * kernel_b_stride + i_c_in * kernel_c_stride + (kernel_size_y - y_k - 1) * kernel_y_stride + (kernel_size_x - x_k - 1)] * input_pixel;
                    }
                } 
            }
        }
        v_dest[i_b * stride_batch_out + i_c_out * stride_c_out + stride_y_out *  global_id.y +  global_id.x] = sum;
    }
}