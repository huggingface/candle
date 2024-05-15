struct MetaUnary{
    operation : u32,
    length : u32,
    scalar1 : u32, //optionally scalar value
    scalar2 : u32
}

struct MetaBinaryScalar{
    operation : u32,
    length : u32,
    scalar : u32
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
const MAX_STRIDE_SIZE = 4; 
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
    p1 : u32, 
}


@group(0) @binding(0)
var<storage, read_write> v_dest: array<u32>;

@group(0) @binding(0) 
var<storage, read_write> v_dest_u32: array<u32>; //Output for U8, and U32

//@group(0) @binding(0) 
//var<storage, read_write> v_dest_f16: array<f16>; //Output for f16

@group(0) @binding(1)
var<uniform> op_unary : MetaUnary;

@group(0) @binding(1)
var<uniform> op_unary_scalar : MetaBinaryScalar;

@group(0) @binding(1)
var<uniform> op_matmul : MetaInfoMatMul;

@group(0) @binding(1)
var<uniform> op_input_matrix : array<MatrixLayout, 3>;

@group(0) @binding(2)
var<storage> v_input1: array<u32>;

@group(0) @binding(3)
var<storage> v_input2: array<u32>;

fn get_shape(m : MatrixLayout, index : u32) -> u32{
    switch index{
        case 0u {return m.shape1;}
        case 1u {return m.shape2;}
        case 2u {return m.shape3;}
        case 3u {return m.shape4;}
        case 4u {return m.shape5;}
        default : {return m.shape1;}
    }
}

fn get_stride(m : MatrixLayout, index : u32) -> u32{
    switch index{
        case 0u {return m.stride1;}
        case 1u {return m.stride2;}
        case 2u {return m.stride3;}
        case 3u {return m.stride4;}
        case 4u {return m.stride5;}
        default : {return m.stride1;}
    }
}

fn set_output_u32(m : u32, value : u32){
    v_dest_u32[m] = value;
}

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


fn rand_uniform(value: u32) -> f32 {
    // Use XORShift algorithm to generate a pseudo-random float
    // Parameters for XORShift algorithm (adjust as needed)
    var state: u32 = value ^ 0x5F3759DF; // Initial state, can be any non-zero value
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    // Convert u32 to a float between 0 and 1
    // Divide by maximum u32 value to get a float in [0, 1)
    return f32(state) / f32(0xFFFFFFFFu);
}


// Function to convert a uniformly distributed random number [0, 1) to a normal distribution with mean and std
fn uniform_to_normal(mean: f32, std_: f32, u1: f32, u2 : f32) -> f32 {
    // Box-Muller transform to convert uniform random value to normal distribution
    let pi: f32 = 3.141592653589793;    

    let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
    let z = mean + std_ * z0; // Convert to desired mean and std
    return z;
}

// Function to convert a uniformly distributed random number [0, 1) to a normal distribution with mean and std
fn rand_normal(id: u32, mean: f32, std_: f32) -> f32 {
    let u1 = rand_uniform(id);
    let u2 = rand_uniform(id + 1);
    return uniform_to_normal(mean, std_, u1, u2);
}



// fn rand(vec2 co) -> f32{
//     return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
// }

const SQRT_TWO_OVER_PI_F32: f32 = 0.79788456080286535587989211986876373;

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
        }case 36u{  //Binary Step
            if(v_dest[id]) < 0{
                v_dest[id] = 0u;
            }
            else{
                v_dest[id] = 1u;
            }
        }case 38u{   //Relu
            v_dest[id] = max(0u, x) ;
        }
        case 43u{ //Identity
            v_dest[id] = x;
        }
        case 44u{ //square
            v_dest[id] = x * x;
        }
        // case 47u{ //random_normal
        //     v_dest[id] = rand_normal(id, scalar1, scalar2);
        // }
        // case 48u{ //random_normal
        //     let r = rand_uniform(id);
        //     v_dest[id] = (scalar2 - scalar1) * r + scalar1;
        // }
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
    if(id >= op_unary.length){
        return;
    }
    let x = v_dest[id];
    set_unary(op_unary.operation,id, x, op_unary.scalar1, op_unary.scalar2);
}


@compute
@workgroup_size(64,1,1)
fn unary_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if(id >= op_unary.length){
        return;
    }

    let x = v_input1[id];
    set_unary(op_unary.operation,id, x, op_unary.scalar1, op_unary.scalar2);
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
        // case 7u{ //powf
        //     v_dest[id] = pow(x, y);
        // }
        default{

        }
    }
}


@compute
@workgroup_size(64,1,1)
fn binary_buffer_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if(id >= op_unary.length){
        return;
    }
    set_binary(op_unary.operation, id, v_dest[id], v_input1[id]);
}

@compute
@workgroup_size(64,1,1)
fn binary_buffer_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if(id >= op_unary.length){
        return;
    }
    set_binary(op_unary.operation, id, v_input1[id], v_input2[id]);
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


@compute
@workgroup_size(1,1,1)
fn reduce_from_buffer() {
    // let id = global_id.x;
    // if(id >= op_unary.length){
    //     return;
    // }

    switch(op_unary.operation){
        case 0u{ //sum
            var sum = 0u;
            for (var i = 0u; i < op_unary.length; i++){
                sum += v_input1[i];
            }
            v_dest[0] = sum;
        }
        case 1u{ //min
            var sum = v_input1[0];
            for (var i = 0u; i < op_unary.length; i++){
                sum = min(sum,v_input1[i]);
            }
            v_dest[0] = sum;
        }
        case 2u{ //max
            var sum = v_input1[0];
            for (var i = 0u; i < op_unary.length; i++){
                sum = max(sum,v_input1[i]);
            }
            v_dest[0] = sum;
        }
        case 3u{//ArgMin
            var sum = v_input1[0];
            var index = 0u;
            for (var i = 0u; i < op_unary.length; i++){
                if v_input1[i] < sum{
                    sum = v_input1[i];
                    index = i;
                }
            }
            v_dest[0] = index;
        }
        case 4u{//ArgMax
            var sum = v_input1[0];
            var index = 0u;
            for (var i = 0u; i < op_unary.length; i++){
                if v_input1[i] > sum{
                    sum = v_input1[i];
                    index = i;
                }
            }
            v_dest[0] = index;
        }
        default{

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
    if(id >= op_unary.length){
        return;
    }
   
    switch(op_unary.operation){
        case 0u: { //eq
            v_dest_u32[id] = bool_to_int(v_input1[id] == v_input2[id]);
        }
        case 1u: {//ne
            v_dest_u32[id] = bool_to_int(v_input1[id] != v_input2[id]);
        }
        case 2u: {//lt
            v_dest_u32[id] = bool_to_int(v_input1[id] < v_input2[id]);
        }
        case 3u: {//LE
            v_dest_u32[id] = bool_to_int(v_input1[id] <= v_input2[id]);
        }
        case 4u: {//GT
            v_dest_u32[id] = bool_to_int(v_input1[id] > v_input2[id]);
        }
        case 5u: {//GE
            v_dest_u32[id] = bool_to_int(v_input1[id] >= v_input2[id]);
        }
        default:{
            
        }
    }
}





// @compute
// @workgroup_size(64,1,1)
// fn copy_stride(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     let x = global_id.x;

//     let info_dest   = op_input_matrix[0];
//     let info_input1 = op_input_matrix[1];

//     v_dest[info_dest.offset + x] = v_input1[info_input1.offset + x];
// }


