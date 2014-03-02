import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getAlignKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif

        #define Conc(x,y) a[x*D1 + y]
        #define Dir(x,y) b[x*D1 + y]

        #define Newdir(x,y) out[x*D1 + y]

        __kernel void align(
                      __global const $nType *a,
                      __global const $nType *b,
                      __global $nType *out,
                      const int D1,
                      const int D2,
                      const $nType attraction,
                      const $nType step) {

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

        $nType direction = Dir(i, j);

        $nType diff[3][3];
        $nType potential[3][3];

        diff[2][0] = fmod(direction - Dir(north,west), M_PI);
        diff[2][1] = fmod(direction - Dir(north,j   ), M_PI);
        diff[2][2] = fmod(direction - Dir(north,east), M_PI);
        diff[1][0] = fmod(direction - Dir(i,west    ), M_PI);
        diff[1][1] = 0;
        diff[1][2] = fmod(direction - Dir(i,east    ), M_PI);
        diff[0][0] = fmod(direction - Dir(south,west), M_PI);
        diff[0][1] = fmod(direction - Dir(south,j   ), M_PI);
        diff[0][2] = fmod(direction - Dir(south,east), M_PI);

        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                potential[i][j] = -sin( 2 * diff[i][j] );
            }
        }


        $nType dtheta = 0;
        $nType concentration = Conc(i,j);

        dtheta += concentration * Conc(north,west) * potential[2][0];
        dtheta += concentration * Conc(north,j   ) * potential[2][1];
        dtheta += concentration * Conc(north,east) * potential[2][2];
        dtheta += concentration * Conc(i,west    ) * potential[1][0];

        dtheta += concentration * Conc(i,east    ) * potential[1][2];
        dtheta += concentration * Conc(south,west) * potential[0][0];
        dtheta += concentration * Conc(south,j   ) * potential[0][1];
        dtheta += concentration * Conc(south,east) * potential[0][2];

        dtheta = attraction * dtheta / 8;
        $nType ndir = direction + dtheta * step;

        Newdir(i,j) = fmod(ndir,M_PI);
    }
    "
end

function alignCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T},
    out_buff :: Buffer{T},
    attraction :: T, step :: T, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "align")

    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, out_buff, int32(d1), int32(d2), attraction, step)
end