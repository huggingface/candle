#include <metal_stdlib>

using namespace metal;

struct fault_counter {
    uint counter;
    uint tolerance;

    fault_counter(uint tolerance) {
        this->counter = 0;
        this->tolerance = tolerance;
    }

    bool quit() {
        counter += 1;
        return (counter > tolerance);
    }
};