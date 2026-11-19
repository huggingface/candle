// Include the file `standard-include.glsl` from one of the standard
// directories.
#include <standard-include.glsl>
// Try to locate the file `relative-include.glsl` in a sibling directory
// and include it.
#include "../relative-shaders/relative-include.glsl"

uint multiply_by_12(in uint arg) {
    return 2 * multiply_by_3(multiply_by_2(arg));
}
