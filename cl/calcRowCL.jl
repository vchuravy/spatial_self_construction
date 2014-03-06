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
        #define number $nType

        #define M(x,y) a[y*D2 + x]
        #define A(x,y) b[y*D2 + x]
        #define F(x,y) c[y*D2 + x]
        #define W(x,y) d[y*D2 + x]

        #define Out(x,y) out[y*D2 + x]

        __kernel void calcRow(
                      __global const number *a,
                      __global const number *b,
                      __global const number *c,
                      __global const number *d,
                      __global number *out,
                      const int D1,
                      const int D2,
                      const number m,
                      const number _a,
                      const number f,
                      const number w) {

        int i = get_global_id(0);
        int j = get_global_id(1);

        Out(i, j) = pow(M(i, j), m) * pow(A(i, j), _a) * pow(F(i, j), f) * pow(W(i, j), w);
    }
"
end

function calcRowCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T}, c_buff :: Buffer{T}, d_buff :: Buffer{T},
    out_buff :: Buffer{T},
    a :: Real, b :: Real, c :: Real, d :: Real, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "calcRow")

    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, c_buff, d_buff, out_buff, int32(d1), int32(d2), convert(T, a), convert(T, b), convert(T, c), convert(T, d))
end