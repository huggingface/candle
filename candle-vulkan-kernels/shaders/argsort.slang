// Last-axis argsort (float input):
// out[row * cols + j] = index (0..cols) of the j-th smallest (asc) or
// j-th largest (desc) element of in[row], i.e. the stable sort order of
// the row. Ties keep their original relative order and NaN values sort
// last in both directions, matching the CPU reference in
// `candle-core/src/sort.rs`.
import common_argsort;
StructuredBuffer<float> g_In;
RWStructuredBuffer<uint> g_Out;
layout(push_constant) Params g_Params;

// One row per work item. The output buffer doubles as the index
// workspace: it is initialized to the identity permutation and sorted
// in place with an O(n log n) heapsort (the previous insertion sort was
// O(n^2)). The (value, index) comparison gives a total order, so ties
// keep their original relative order and NaN sorts last in both
// directions, matching the CPU reference in `candle-core/src/sort.rs`.
//
// `less(base, a, b)` is "element a sorts before element b".
bool less(uint64_t base, uint a, uint b, bool desc)
{
    float va = g_In[base + a];
    float vb = g_In[base + b];
    if (isnan(va)) { return false; }
    if (isnan(vb)) { return true; }
    if (va == vb) { return a < b; }
    return desc ? (va > vb) : (va < vb);
}

void sift_down(uint64_t base, uint root, uint len, bool desc)
{
    while (true)
    {
        uint child = root * 2u + 1u;
        if (child >= len) { break; }
        uint right = child + 1u;
        // Pick the child that sorts later (max in the sort order).
        if (right < len && less(base, g_Out[base + child], g_Out[base + right], desc))
        {
            child = right;
        }
        // If the child sorts before the root, the root is already the max.
        if (less(base, g_Out[base + child], g_Out[base + root], desc)) { break; }
        uint t = g_Out[base + root];
        g_Out[base + root] = g_Out[base + child];
        g_Out[base + child] = t;
        root = child;
    }
}

void argsort_row(uint row, bool desc)
{
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    uint64_t base = uint64_t(row) * cols;
    for (uint64_t j = 0u; j < cols; j += 1u)
    {
        g_Out[base + j] = uint(j);
    }
    uint len = uint(cols);
    // Build the max-heap.
    for (uint i = len / 2u; i > 0u; i -= 1u)
    {
        sift_down(base, i - 1u, len, desc);
    }
    // Repeatedly move the max to the end and restore the heap.
    for (uint end = len - 1u; end > 0u; end -= 1u)
    {
        uint t = g_Out[base];
        g_Out[base] = g_Out[base + end];
        g_Out[base + end] = t;
        sift_down(base, 0u, end, desc);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_argsort_asc(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    argsort_row(dispatchThreadID.x, false);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_argsort_desc(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    argsort_row(dispatchThreadID.x, true);
}
