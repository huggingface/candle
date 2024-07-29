
// struct vec3<T>{
//     pub x : T,
//     pub y : T,
//     pub z : T
// }

// #[derive(Debug)]
// enum test{
// None
// }



// #[workgroup_size(16, 16, 1)]
// fn binary_buffer_from_buffer3d(#[builtin(global_invocation_id)] global_id: vec3<u32>) {
//     let id = global_id.x + global_id.y * 65535 * 64;

//     let input2_layout = op_binary.input1_layout + get_size2(op_binary.dims);

//     let pos1 = get_index2(op_binary.input1_layout, id, op_binary.dims);
//     let pos2 = get_index2(input2_layout, id, op_binary.dims);
    
//     if(pos1.is_valid){
//         set_binary(op_binary.operation, id, v_input1[pos1.id], v_input2[pos2.id]);
//     }
// }

