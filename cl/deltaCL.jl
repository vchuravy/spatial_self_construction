import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getDeltaKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif

        #define R1(x,y) row1[y*D2 + x]
        #define R2(x,y) row2[y*D2 + x]
        #define R3(x,y) row3[y*D2 + x]
        #define R4(x,y) row4[y*D2 + x]
        #define R5(x,y) row5[y*D2 + x]
        #define R6(x,y) row6[y*D2 + x]

        #define Out(x,y) out[y*D2 + x]

        __kernel void delta(
                      __global const $nType *row1,
                      __global const $nType *row2,
                      __global const $nType *row3,
                      __global const $nType *row4,
                      __global const $nType *row5,
                      __global const $nType *row6,
                      __global $nType *out,
                      const $nType r1,
                      const $nType r2,
                      const $nType r3,
                      const $nType r4,
                      const $nType r5,
                      const $nType r6,
                      const int D1,
                      const int D2) {

        int i = get_global_id(0);
        int j = get_global_id(1);

        Out(i,j) = r1 * R1(i,j) + r2 * R2(i,j) + r3 * R3(i,j) + r4 * R4(i,j) + r5 * R5(i,j) + r6 * R6(i,j);
    }
"
end

function deltaCL!{T <: FloatingPoint}(
    row1_buff :: Buffer{T}, row2_buff :: Buffer{T}, row3_buff :: Buffer{T}, row4_buff :: Buffer{T}, row5_buff :: Buffer{T}, row6_buff :: Buffer{T},
    out_buff :: Buffer{T},
    r1 :: T, r2 :: T, r3 :: T , r4 :: T, r5 :: T, r6 :: T,
    d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "delta")

    cl.call(queue, k, (d1,d2), nothing, row1_buff, row2_buff, row3_buff, row4_buff, row5_buff, row6_buff, out_buff, r1, r2, r3, r4, r5, r6, int32(d1), int32(d2))
end