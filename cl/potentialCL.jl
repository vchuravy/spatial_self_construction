import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

const potentialKernel = "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #define PI M_PI
        #define PI_4 M_PI_4
        #define number double
        #define number4 double4
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #define PI M_PI
        #define PI_4 M_PI_4
        #define number double
        #define number4 double4
        #else
        #define PI M_PI
        #define PI_4 M_PI_4_F
        #define number float
        #define number4 float4
        #endif

        #define Afield(x,y) a[y*D2 + x]
        #define Bfield(x,y) b[y*D2 + x]
        #define Area(x,y)   d[y*D2 + x]

        #define Apot(x,y) aout[y*D2 + x]
        #define Bpot(x,y) bout[y*D2 + x]

        __kernel void area(
                      __global const number *d,
                      __global number4 *dout,
                      const number LONG,
                      const number SHORT) {

            int i = get_global_id(0);

            const number dir = d[i];

            const number area1 = LONG*SHORT/2 * (   PI/8-dir - atan((SHORT-LONG)*sin(2*(  PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(  PI/8-dir)))));
            const number area2 = LONG*SHORT/2 * ( 3*PI/8-dir - atan((SHORT-LONG)*sin(2*(3*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(3*PI/8-dir)))));
            const number area3 = LONG*SHORT/2 * ( 5*PI/8-dir - atan((SHORT-LONG)*sin(2*(5*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(5*PI/8-dir)))));
            const number area4 = LONG*SHORT/2 * ( 7*PI/8-dir - atan((SHORT-LONG)*sin(2*(7*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(7*PI/8-dir)))));
            const number area5 = LONG*SHORT/2 * ( 9*PI/8-dir - atan((SHORT-LONG)*sin(2*(9*PI/8-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(9*PI/8-dir)))));

            const number lps = SHORT * LONG * PI;

            number4 a;

            a.s0 = (area2-area1)/lps;
            a.s1 = (area3-area2)/lps;
            a.s2 = (area4-area3)/lps;
            a.s3 = (area5-area4)/lps;

            dout[i] = a;
        }

        __kernel void potential(
                      __global const number *a,
                      __global const number *b,
                      __global const number4 *d,
                      __global number *aout,
                      __global number *bout,
                      const int D1,
                      const int D2,
                      const number repulsion) {

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

            const number4 a_ij = Area(i, j);

            const number r_ij = repulsion * Bfield(i,j);

            const number a_ne = Area(north, east).s0;
            const number a_nj = Area(north, j   ).s1;
            const number a_nw = Area(north, west).s2;
            const number a_iw = Area(i    , west).s3;
            const number a_sw = Area(south, west).s0;
            const number a_sj = Area(south, east).s1;
            const number a_se = Area(south, j   ).s2;
            const number a_ie = Area(i    , east).s3;

            const number r_ne = repulsion * Bfield(north, east);
            const number r_nj = repulsion * Bfield(north, j   );
            const number r_nw = repulsion * Bfield(north, west);
            const number r_iw = repulsion * Bfield(i    , west);
            const number r_sw = repulsion * Bfield(south, west);
            const number r_sj = repulsion * Bfield(south, east);
            const number r_se = repulsion * Bfield(south, j   );
            const number r_ie = repulsion * Bfield(i    , east);

            Apot(i,j) =    fma(r_ne, a_ne,
                           fma(r_nj, a_nj,
                           fma(r_nw, a_nw,
                           fma(r_iw, a_iw,
                           fma(r_sw, a_sw,
                           fma(r_sj, a_sj,
                           fma(r_se, a_se,
                           fma(r_ie, a_ie, r_ij))))))));

            Bpot(i,j) = repulsion *     fma(Afield(north,east), a_ij.s0,
                                        fma(Afield(north,j   ), a_ij.s1,
                                        fma(Afield(north,west), a_ij.s2,
                                        fma(Afield(i,west    ), a_ij.s3,
                                        fma(Afield(south,west), a_ij.s0,
                                        fma(Afield(south,j   ), a_ij.s1,
                                        fma(Afield(south,east), a_ij.s2,
                                        fma(Afield(i,east    ), a_ij.s3, Afield(i,j)))))))));
    }
"

function potentialCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T}, d_buff :: Buffer{T},
    aout_buff :: Buffer{T}, bout_buff :: Buffer{T},
    repulsion :: T, long :: T, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    area_buff = cl.Buffer(T, ctx, :rw, d1 * d2 * 4)

    k_p = cl.Kernel(program, "potential")
    k_a = cl.Kernel(program, "area")

    short = one(T)
    cl.call(queue, k_a, d1*d2, nothing, d_buff, area_buff, long, short)

    cl.call(queue, k_p, (d1,d2), nothing, a_buff, b_buff, area_buff, aout_buff, bout_buff, int32(d1), int32(d2), repulsion)
end