import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getSMulKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif

        #define number $nType

        __kernel void smul(
                      const number s,
                      __global const number *a,
                      __global number *out) {

        int i = get_global_id(0);

        out[i] = s * a[i];
    }
"
end

function smulCL!{T <: FloatingPoint}(
    s :: Real, a_buff :: Buffer{T},
    out_buff :: Buffer{T},
    d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "smul")

    cl.call(queue, k, d1 * d2, nothing, convert(T, s), a_buff, out_buff)
end