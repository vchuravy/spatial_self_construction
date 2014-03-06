import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getLaplacianKernel{T <: FloatingPoint}(:: Type{T})
        nType = T == Float64 ? "double" : "float"

        return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif
        #define number $nType

        #define Ap(x,y) a[y*D2 + x]

        #define A_lap(x,y) out[y*D2 + x]

        __kernel void laplacian(
                      __global const number *a,
                      __global number *out,
                      const int D1,
                      const int D2) {

            int i = get_global_id(0);
            int j = get_global_id(1);
            int west = j-1;
            int east = j+1;
            int north = i+1;
            int south = i-1;

            if(j == 0)
                west = D2 - 1;
            if(j == D2 - 1)
                east = 0;
            if(i == D1 - 1)
                north = 0;
            if(i == 0)
                south = D1 - 1;

            // Calulate the laplacian

            A_lap(i,j) = -Ap(i,j) +(Ap(north, east) +
                                    Ap(north, j   ) +
                                    Ap(north, west) +
                                    Ap(i    , west) +
                                    Ap(south, west) +
                                    Ap(south, j   ) +
                                    Ap(south, east) +
                                    Ap(i    , east))/8;
    }
"
end

function laplacianCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T},
    out_buff :: Buffer{T},
    d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "laplacian")

    cl.call(queue, k, (d1,d2), nothing, a_buff, out_buff, int32(d1), int32(d2))
end