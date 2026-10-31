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
// in place with a stable insertion sort, so the final buffer content is
// the argsort result.
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
    for (uint64_t j = 1u; j < cols; j += 1u)
    {
        uint cur = g_Out[base + j];
        float cv = g_In[base + cur];
        uint64_t k = j;
        while (k > 0u)
        {
            uint prev = g_Out[base + k - 1u];
            float pv = g_In[base + prev];
            bool move;
            if (isnan(cv))
            {
                // NaN never moves: it sorts last in both directions and
                // keeps its original order relative to other NaNs.
                move = false;
            }
            else if (isnan(pv))
            {
                move = true;
            }
            else if (desc)
            {
                move = cv > pv;
            }
            else
            {
                move = cv < pv;
            }
            if (!move)
            {
                break;
            }
            g_Out[base + k] = g_Out[base + k - 1u];
            k -= 1u;
        }
        g_Out[base + k] = cur;
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
