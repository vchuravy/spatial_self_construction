import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getDelta2Kernel{T <: FloatingPoint}(:: Type{T})
        nType = T == Float64 ? "double" : "float"

        return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif
        #define number $nType

        #define delta(x,y)  d[y*D2 + x]
        #define AField(x,y) a[y*D2 + x]
        #define BField(x,y) b[y*D2 + x]
        #define Lap(x,y)    l[y*D2 + x]

        #define Out(x,y)    out[y*D2 + x]

        __kernel void delta2(
                      __global const number *d,
                      __global const number *a,
                      __global const number *b,
                      __global const number *l,
                      __global number *out,
                      const int D2,
                      const number decay) {

            int i = get_global_id(0);
            int j = get_global_id(1);

            Out(i,j) = delta(i,j) / (1 + AField(i,j)) - decay * BField(i,j) + Lap(i,j);
        }
"
end

function delta2CL!{T <: FloatingPoint}(
    delta_buff :: Buffer{T}, af_buff :: Buffer{T}, bf_buff :: Buffer{T}, lap_buff :: Buffer{T},
    out_buff :: Buffer{T},
    decay :: Real, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "delta2")

    cl.call(queue, k, (d1,d2), nothing, delta_buff, af_buff, bf_buff, lap_buff, out_buff, int32(d2), convert(T, decay))
end