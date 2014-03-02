import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getPotentialKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif

        #define Afield(x,y) a[x*D1 + y]
        #define Bfield(x,y) b[x*D1 + y]
        #define Direction(x,y) d[x*D1 + y]

        #define Apot(x,y) aout[x*D1 + y]
        #define Bpot(x,y) bout[x*D1 + y]

        $nType area($nType x, $nType d, $nType LONG, $nType SHORT) {
            const $nType p8 = x * M_PI/8;
            return LONG * SHORT / 2 * ( p8 - d - atan((SHORT - LONG)*sin(2*(p8 - d)) / (SHORT + LONG + (SHORT - LONG)*cos(2*(p8 - d)))));
        }

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
        const $nType bF = Bfield(i,j);
        const $nType cRepulsion = selfRepulsion * bF;

        Apot(i,j) += cRepulsion;

        const $nType area1 = area(1.0f, dir, LONG, SHORT);
        const $nType area2 = area(3.0f, dir, LONG, SHORT);
        const $nType area3 = area(5.0f, dir, LONG, SHORT);
        const $nType area4 = area(7.0f, dir, LONG, SHORT);
        const $nType area5 = area(9.0f, dir, LONG, SHORT);

        const $nType lps = SHORT * LONG * M_PI;

        const $nType a1 = (area2-area1)/lps;
        const $nType a2 = (area3-area2)/lps;
        const $nType a3 = (area4-area3)/lps;
        const $nType a4 = (area5-area4)/lps;

        Apot(north,east) += cRepulsion*a1;
        Apot(north,j   ) += cRepulsion*a2;
        Apot(north,west) += cRepulsion*a3;
        Apot(i,west    ) += cRepulsion*a4;

        Apot(south,west) += cRepulsion*a1;
        Apot(south,j   ) += cRepulsion*a2;
        Apot(south,east) += cRepulsion*a3;
        Apot(i,east    ) += cRepulsion*a4;

        Bpot(i,j) += selfRepulsion * ( Afield(i,j)
                                            + Afield(north,east)  * a1
                                            + Afield(north,j   )  * a2
                                            + Afield(north,west)  * a3
                                            + Afield(i,west    )  * a4
                                            + Afield(south,west)  * a1
                                            + Afield(south,j   )  * a2
                                            + Afield(south,east)  * a3
                                            + Afield(i,east    )  * a4
                                        );
    }
"
end

function potentialCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T}, d_buff :: Buffer{T},
    aout_buff :: Buffer{T}, bout_buff :: Buffer{T},
    repulsion :: T, long :: T, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    zeroArray = zeros(T, d1, d2)

    cl.copy!(queue, aout_buff, zeroArray)
    cl.copy!(queue, bout_buff, zeroArray)

    k = cl.Kernel(program, "potential")

    short = one(T)
    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, d_buff, aout_buff, bout_buff, int32(d1), int32(d2), repulsion, long, short)
end