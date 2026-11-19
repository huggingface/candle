#version 450
#extension GL_EXT_mesh_shader : require

// In mesh shaders you have to load all data manually from storage buffers, which are declared just like uniform
// buffers, but using the `buffer` keyword. You may not use:
// * `in`: Unlike vertex shaders, Mesh shaders do not have an input assembly (IA) stage that pulls data from buffers
//    and forwards them to the vertex shaders as `in` inputs.
// * `uniform`: Uniform buffers have to be of constant size, but as our buffers may have a varying amount of data,
//    they have to be storage buffers instead.
//
// The triangle vertex positions.
layout(set = 0, binding = 0) buffer VertexBuffer {
    vec2 position[];
} buffer_vertex;

// The per-instance data.
struct Instance {
    vec2 position_offset;
    float scale;
};

layout(set = 0, binding = 1) buffer InstanceBuffer {
    Instance instance[];
} buffer_instance;

// This declaration specifies the workgroup size of the mesh shader, similarly to compute shaders
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
// This declares the type of primitive you want to emit, typically triangles, as well as maximum amount of vertices
// and primitives you may emit. Primitives may only be in lists, aka. triangle_strip or triangle_fan are not allowed.
layout(triangles, max_vertices = 3, max_primitives = 1) out;

// As mesh shaders may emit multiple vertices, all outputs have to be an array. See below, when vertices are emitted.
layout(location = 0) out vec4 out_color[];

const uint rows = 10;
const uint cols = 10;
const uint n_instances = rows * cols;

void main() {
    vec2 position_offset;
    float scale;
    vec4 color;

    // There are two main use-cases for mesh shaders, switch in between them here.
    // They should both draw the same triangles, but with different colors.
    const bool LOAD_FROM_INSTANCE_BUFFER = false;

    if (LOAD_FROM_INSTANCE_BUFFER) {
        // Use-case 1: load instance data from buffers, similarly to doing an instanced draw
        // color triangles red
        color = vec4(1.0, 0.0, 0.0, 1.0);

        Instance instance = buffer_instance.instance[gl_GlobalInvocationID.y * rows + gl_GlobalInvocationID.x];
        position_offset = instance.position_offset;
        scale = instance.scale;

    } else {
        // Use-case 2: generate the geometry dynamically in the mesh shader
        // color triangles green
        color = vec4(0.0, 1.0, 0.0, 1.0);

        uint c = gl_GlobalInvocationID.x;
        uint r = gl_GlobalInvocationID.y;

        // the same algo for generating the triangle data as in the instanced example
        float half_cell_w = 0.5 / float(cols);
        float half_cell_h = 0.5 / float(rows);
        float x = half_cell_w + (c / float(cols)) * 2.0 - 1.0;
        float y = half_cell_h + (r / float(rows)) * 2.0 - 1.0;
        position_offset = vec2(x, y);
        scale = (2.0 / float(rows)) * (c * float(rows) + r) / n_instances;
    }

    // Dynamically set the amount of vertices and triangles that you would like to emit, must be lower than what was
    // declared above. From the `OpSetMeshOutputsEXT` spec:
    // The arguments are taken from the first invocation in each workgroup. Behavior is undefined if any invocation
    // executes this instruction more than once or under non-uniform control flow. Behavior is undefined if there is
    // any control flow path to an output write that is not preceded by this instruction.
    SetMeshOutputsEXT(
        3, // vertices
        1// triangles = indices / 3
    );

    // emit vertex data
    for (uint i = 0; i < 3; i++) {
        // As we may emit multiple vertices, all outputs are arrays. You index into them using a unique vertex index
        // within your work group. In this example the work group has the size (1, 1, 1), so each invocation can
        // simply use the indices [0-2]. With larger work groups you will have to use the `gl_LocalInvocationID` to
        // compute indices and make sure they are unique, so results don't get overridden by other invocations.
        out_color[i] = color;
        // just like setting gl_Position in the vertex shader
        gl_MeshVerticesEXT[i].gl_Position = vec4(buffer_vertex.position[i] * scale + position_offset, 0.0, 1.0);
    }

    // emit triangle indices
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
