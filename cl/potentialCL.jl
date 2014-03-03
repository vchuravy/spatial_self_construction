import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getPotentialKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #define PI M_PI
        #define PI_4 M_PI_4
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #define PI M_PI
        #define PI_4 M_PI_4_F
        #else
        #define PI M_PI
        #define PI_4 M_PI_4_F
        #endif

        #define Afield(x,y) a[y*D2 + x]
        #define Bfield(x,y) b[y*D2 + x]
        #define Direction(x,y) d[y*D2 + x]

        #define Apot(x,y) aout[y*D2 + x]
        #define Bpot(x,y) bout[y*D2 + x]

        __kernel void potential(
                      __global const $nType *a,
                      __global const $nType *b,
                      __global const $nType *d,
                      __global $nType *aout,
                      __global $nType *bout,
                      const int D1,
                      const int D2,
                      const $nType selfRepulsion,
                      const $nType LONG,
                      const $nType SHORT) {

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

        const $nType dir = Direction(i,j);
        const $nType cRepulsion = selfRepulsion * Bfield(i,j);

        Apot(i,j) += cRepulsion;


        const $nType area1 = LONG*SHORT/2 * (   PI/8-dir - atan((SHORT-LONG)*sin(2*(  PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(  PI/8-dir)))));
        const $nType area2 = LONG*SHORT/2 * ( 3*PI/8-dir - atan((SHORT-LONG)*sin(2*(3*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(3*PI/8-dir)))));
        const $nType area3 = LONG*SHORT/2 * ( 5*PI/8-dir - atan((SHORT-LONG)*sin(2*(5*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(5*PI/8-dir)))));
        const $nType area4 = LONG*SHORT/2 * ( 7*PI/8-dir - atan((SHORT-LONG)*sin(2*(7*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(7*PI/8-dir)))));
        const $nType area5 = LONG*SHORT/2 * ( 9*PI/8-dir - atan((SHORT-LONG)*sin(2*(9*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(9*PI/8-dir)))));

        const $nType lps = SHORT * LONG * PI;

        const $nType a1 = (area2-area1)/lps;
        const $nType a2 = (area3-area2)/lps;
        const $nType a3 = (area4-area3)/lps;
        const $nType a4 = (area5-area4)/lps;

        Apot(north,east) = fma(cRepulsion, a1, Apot(north,east));
        Apot(north,j   ) = fma(cRepulsion, a2, Apot(north,j   ));
        Apot(north,west) = fma(cRepulsion, a3, Apot(north,west));
        Apot(i,west    ) = fma(cRepulsion, a4, Apot(i,west    ));

        Apot(south,west) = fma(cRepulsion, a1, Apot(south,west));
        Apot(south,j   ) = fma(cRepulsion, a2, Apot(south,j   ));
        Apot(south,east) = fma(cRepulsion, a3, Apot(south,east));
        Apot(i,east    ) = fma(cRepulsion, a4, Apot(i    ,east));


        Bpot(i,j) = selfRepulsion * fma(Afield(north,east), a1,
                                    fma(Afield(north,j   ), a2,
                                    fma(Afield(north,west), a3,
                                    fma(Afield(i,west    ), a4,
                                    fma(Afield(south,west), a1,
                                    fma(Afield(south,j   ), a2,
                                    fma(Afield(south,east), a3,
                                    fma(Afield(i,east    ), a4, Afield(i,j)))))))));
    }
"
end

function potentialCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T}, d_buff :: Buffer{T},
    aout_buff :: Buffer{T}, bout_buff :: Buffer{T},
    repulsion :: T, long :: T, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    cl.copy!(queue, aout_buff, zeros(T, d1, d2))
    #cl.copy!(queue, bout_buff, zeroArray)

    k = cl.Kernel(program, "potential")

    short = one(T)
    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, d_buff, aout_buff, bout_buff, int32(d1), int32(d2), repulsion, long, short)
end