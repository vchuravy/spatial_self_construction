import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getRowKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif

        #define M(x,y) a[x*D1 + y]
        #define A(x,y) b[x*D1 + y]
        #define F(x,y) c[x*D1 + y]
        #define W(x,y) d[x*D1 + y]

        #define Out(x,y) out[x*D1 + y]

        __kernel void calcRow(
                      __global const $nType *a,
                      __global const $nType *b,
                      __global const $nType *c,
                      __global const $nType *d,
                      __global $nType *out,
                      const int D1,
                      const int D2,
                      const int m,
                      const int _a,
                      const int f,
                      const int w) {

        int i = get_global_id(0);
        int j = get_global_id(1);

        Out(i, j) = pown(M(i, j), m) * pown(A(i, j), _a) * pown(F(i, j), f) * pown(W(i, j), w);
    }
"
end

function calcRowCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T}, c_buff :: Buffer{T}, d_buff :: Buffer{T},
    out_buff :: Buffer{T},
    a :: Real, b :: Real, c :: Real, d :: Real, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "calcRow")

    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, c_buff, d_buff, out_buff, int32(d1), int32(d2), int32(a), int32(b), int32(c), int32(d))
end