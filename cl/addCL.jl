function getAddKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif
        #define number $nType

        __kernel void add(
                      __global const number *a,
                      __global const number *b,
                      __global number *out) {

        int i = get_global_id(0);

        out[i] = a[i] + b[i];
    }
"
end

function addCL!(
    a_buff, b_buff,
    out_buff,
    d1 :: Int64, d2 :: Int64,
    ctx, queue, program)

    k = Kernel(program, "add")

    call(queue, k, d1 * d2, nothing, a_buff, b_buff, out_buff)
end