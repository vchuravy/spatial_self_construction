import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getPotentialKernel{T <: FloatingPoint}(:: Type{T})
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

            const number selfRepulsion = 1*repulsion; // making this lower than repulsion allows for neighbors to have relative potential, so increases the chance that the hydrophobe will flow.
            const number r_ij = selfRepulsion * Afield(i,j);

            const number a_ne = Area(north, east).s0;
            const number a_nj = Area(north, j   ).s1;
            const number a_nw = Area(north, west).s2;
            const number a_iw = Area(i    , west).s3;
            const number a_sw = Area(south, west).s0;
            const number a_sj = Area(south, j   ).s1;
            const number a_se = Area(south, east).s2;
            const number a_ie = Area(i    , east).s3;

            const number r_ne = repulsion * Afield(north, east);
            const number r_nj = repulsion * Afield(north, j   );
            const number r_nw = repulsion * Afield(north, west);
            const number r_iw = repulsion * Afield(i    , west);
            const number r_sw = repulsion * Afield(south, west);
            const number r_sj = repulsion * Afield(south, j   );
            const number r_se = repulsion * Afield(south, east);
            const number r_ie = repulsion * Afield(i    , east);

            Bpot(i,j) =    fma(r_ne, a_ne,
                           fma(r_nj, a_nj,
                           fma(r_nw, a_nw,
                           fma(r_iw, a_iw,
                           fma(r_sw, a_sw,
                           fma(r_sj, a_sj,
                           fma(r_se, a_se,
                           fma(r_ie, a_ie, r_ij))))))));

            Apot(i,j) = repulsion *     fma(Bfield(north,east), a_ij.s0,
                                        fma(Bfield(north,j   ), a_ij.s1,
                                        fma(Bfield(north,west), a_ij.s2,
                                        fma(Bfield(i,west    ), a_ij.s3,
                                        fma(Bfield(south,west), a_ij.s0,
                                        fma(Bfield(south,j   ), a_ij.s1,
                                        fma(Bfield(south,east), a_ij.s2,
                                        fma(Bfield(i,east    ), a_ij.s3, Bfield(i,j)))))))));
    }
"
end

function potentialCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T}, d_buff :: Buffer{T},
    aout_buff :: Buffer{T}, bout_buff :: Buffer{T},
    repulsion :: Real, long :: Real, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    area_buff = cl.Buffer(T, ctx, :rw, d1 * d2 * 4)

    k_p = cl.Kernel(program, "potential")
    k_a = cl.Kernel(program, "area")

    short = one(T)
    cl.call(queue, k_a, d1*d2, nothing, d_buff, area_buff, convert(T, long), short)

    cl.call(queue, k_p, (d1,d2), nothing, a_buff, b_buff, area_buff, aout_buff, bout_buff, int32(d1), int32(d2), convert(T, repulsion))
end