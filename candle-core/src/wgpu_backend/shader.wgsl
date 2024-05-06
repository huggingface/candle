struct MetaUnary{
    operation : u32,
    length : u32
}

struct MetaBinaryScalar{
    operation : u32,
    length : u32,
    scalar : f32
}

//(M X N) * (N X K)
struct MetaInfoMatMul{
    m : u32, 
    n : u32, 
    k : u32
}



@group(0) @binding(0)
var<storage, read_write> v_dest: array<f32>;

@group(0) @binding(1)
var<uniform> op_unary : MetaUnary;

@group(0) @binding(1)
var<uniform> op_unary_scalar : MetaBinaryScalar;

@group(0) @binding(1)
var<uniform> op_matmul : MetaInfoMatMul;

@group(0) @binding(2)
var<storage> v_input1: array<f32>;

@group(0) @binding(3)
var<storage> v_input2: array<f32>;


//all Unary Operations(No Input)
//TODO SQR, SQRT?
fn set_unary(operation : u32, id : u32, x : f32){

    switch operation {
        case 0u{ // set 0
            v_dest[id] = 0.0;
        }
        case 1u{ //set 1
            v_dest[id] = 1.0;
        }

        //inplace operators
        case 2u{ 
            v_dest[id] += 1.0;
        }
        case 3u{ 
            v_dest[id] -= 1.0;
        }
        case 4u{  
            v_dest[id] = abs(x);
        }
        case 5u{ 
            v_dest[id] = acos(x); 
        }
        case 6u{ 
            v_dest[id] = acosh(x); 
        }
        case 7u{ 
            v_dest[id] = asin(x); 
        }
        case 8u{ 
            v_dest[id] = asinh(x); 
        }
        case 9u{ 
            v_dest[id] = atan(x) ;
        }
        case 10u{ 
            v_dest[id] = atanh(x) ;
        }
        case 11u{ 
            v_dest[id] = ceil(x) ;
        }
        case 12u{ 
            v_dest[id] = cos(x); 
        }
        case 13u{ 
            v_dest[id] = cosh(x) ;
        }
        case 17u{ 
            v_dest[id] = degrees(x) ;
        }
       
        
        case 21u{ 
            v_dest[id] = exp(x); 
        }
        case 22u{ 
            v_dest[id] = floor(x); 
        }
        case 23u{ 
            v_dest[id] = fract(x); 
        }
        case 24u{ 
            v_dest[id] = inverseSqrt(x) ;
        }
        case 25u{ 
            v_dest[id] = log(x) ;
        }
        case 26u{ 
            v_dest[id] = log2(x) ;
        }
        case 27u{ 
            v_dest[id] = radians(x) ;
        }
        case 28u{ 
            v_dest[id] = sign(x) ;
        }
        case 29u{ 
            v_dest[id] = sin(x) ;
        }case 31u{ 
            v_dest[id] = sinh(x) ;
        }case 32u{ 
            v_dest[id] = sqrt(x) ;
        }case 33u{ 
            v_dest[id] = tan(x); 
        }case 34u{ 
            v_dest[id] = tanh(x) ;
        }case 35u{ 
            v_dest[id] = trunc(x) ;
        }case 36u{  //Binary Step
            if(v_dest[id]) < 0{
                v_dest[id] = 0.0;
            }
            else{
                v_dest[id] = 1.0;
            }
        }case 37u{   //Sigmoid
            v_dest[id] = 1 / (1 + exp(-x));
        }case 38u{   //Relu
            v_dest[id] = max(0.0, x) ;
        }case 39u{   //Softplus
            v_dest[id] = log(1 + exp(x));
        }case 40u{  //Leaky ReLU
            if(x) < 0{
                v_dest[id] = 0.01 * x;
            }
            else{
                v_dest[id] = x;
            }
        }case 41u{ //SiLU
            v_dest[id] = x / (1 + exp(-x));
        }
        case 42u{ //Gaussian
            v_dest[id] = exp(-(x * x));
        }
        case 43u{ //Identity
            v_dest[id] = x;
        }
        case 44u{ //square
            v_dest[id] = x * x;
        }
        case 45u{ 
            v_dest[id] = -x;
        }
        case 46u{ //square
            v_dest[id] = 1 / x;
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
    set_unary(op_unary.operation,id, x);
}


@compute
@workgroup_size(64,1,1)
fn unary_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if(id >= op_unary.length){
        return;
    }

    let x = v_input1[id];
    set_unary(op_unary.operation,id, x);
}




fn set_binary(operation : u32, id : u32, x : f32, y : f32){
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
        case 7u{ //powf
            v_dest[id] = pow(x, y);
        }
        default{

        }
    }
}

@compute
@workgroup_size(64,1,1)
fn binary_scalar_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if(id >= op_unary_scalar.length){
        return;
    }
    set_binary(op_unary_scalar.operation, id, v_dest[id], op_unary_scalar.scalar);
}

@compute
@workgroup_size(64,1,1)
fn binary_scalar_from_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if(id >= op_unary_scalar.length){
        return;
    }
    set_binary(op_unary_scalar.operation, id, v_input1[id], op_unary_scalar.scalar);
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
    if(x >= op_matmul.k){
        return;
    }
    if(y >= op_matmul.m){
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < op_matmul.n; i++){
        sum +=  v_input1[y * op_matmul.n + i] * v_input2[i * op_matmul.k + x];
    }

    v_dest[y * op_matmul.k + x] = sum;
}


