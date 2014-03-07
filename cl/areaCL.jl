import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getAreaKernel{T <: FloatingPoint}(:: Type{T})
        nType = T == Float64 ? "double" : "float"
        nPi = T == Float64 ? "M_PI" : "M_PI_F"
        nPi4 = T == Float64 ? "M_PI_4" : "M_PI_4_F"

        return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif
        #define number $nType
        #define number4 $(nType)4
        #define PI $nPi
        #define PI_4 $nPi4

        __kernel void area(
                      __global const number *d,
                      __global number4 *dout,
                      const number LONG,
                      const number SHORT) {

            int i = get_global_id(0);

            const number dir = d[i];

            const number area1 = LONG*SHORT/2 * (   PI_4/2-dir - atan((SHORT-LONG)*sin(2*(  PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(  PI_4/2-dir)))));
            const number area2 = LONG*SHORT/2 * ( 3*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(3*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(3*PI_4/2-dir)))));
            const number area3 = LONG*SHORT/2 * ( 5*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(5*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(5*PI_4/2-dir)))));
            const number area4 = LONG*SHORT/2 * ( 7*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(7*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(7*PI_4/2-dir)))));
            const number area5 = LONG*SHORT/2 * ( 9*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(9*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(9*PI_4/2-dir)))));

            const number lps = SHORT * LONG * PI;

            number4 a;

            a.s0 = (area2-area1)/lps;
            a.s1 = (area3-area2)/lps;
            a.s2 = (area4-area3)/lps;
            a.s3 = (area5-area4)/lps;

            dout[i] = a;
        }
"
end

function areaCL!{T <: FloatingPoint}(
    d_buff :: Buffer{T},
    area_buff :: Buffer{T}, 
    long :: Real,
    d1 :: Int, d2 :: Int, 
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k_a = cl.Kernel(program, "area")

    short = one(T)
    cl.call(queue, k_a, d1*d2, nothing, d_buff, area_buff, convert(T, long), short)
end