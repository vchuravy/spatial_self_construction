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
        #define number8 $(nType)8
        #define PI $nPi
        #define PI_4 $nPi4

        #define Afield(x,y) a[y*D2 + x]
        #define Bfield(x,y) b[y*D2 + x]
        #define Area(x,y)   d[y*D2 + x]

        #define Apot(x,y) aout[y*D2 + x]
        #define Bpot(x,y) bout[y*D2 + x]


        static number sum(const number8 n) {
            return n.s0 + n.s1 +n.s2 + n.s3 + n.s4 + n.s5 + n.s6 + n.s7;
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

            number8 area;
            area.s0 = Area(north, east).s0;
            area.s1 = Area(north, j   ).s1;
            area.s2 = Area(north, west).s2;
            area.s3 = Area(i    , west).s3;
            area.s4 = Area(south, west).s0;
            area.s5 = Area(south, j   ).s1;
            area.s6 = Area(south, east).s2;
            area.s7 = Area(i    , east).s3;

            number8 r_af;
            r_af.s0 = Afield(north, east);
            r_af.s1 = Afield(north, j   );
            r_af.s2 = Afield(north, west);
            r_af.s3 = Afield(i    , west);
            r_af.s4 = Afield(south, west);
            r_af.s5 = Afield(south, j   );
            r_af.s6 = Afield(south, east);
            r_af.s7 = Afield(i    , east);

            r_af = repulsion * r_af;

            Bpot(i,j) =   sum(r_af * area) + r_ij;

            Apot(i,j) = repulsion *(Bfield(north,east) * a_ij.s0 +
                                    Bfield(north,j   ) * a_ij.s1 +
                                    Bfield(north,west) * a_ij.s2 +
                                    Bfield(i,west    ) * a_ij.s3 +
                                    Bfield(south,west) * a_ij.s0 +
                                    Bfield(south,j   ) * a_ij.s1 +
                                    Bfield(south,east) * a_ij.s2 +
                                    Bfield(i,east    ) * a_ij.s3 + Bfield(i,j));
    }
"
end

function potentialCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T}, area_buff :: Buffer{T},
    aout_buff :: Buffer{T}, bout_buff :: Buffer{T},
    repulsion :: Real, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k_p = cl.Kernel(program, "potential")

    cl.call(queue, k_p, (d1,d2), nothing, a_buff, b_buff, area_buff, aout_buff, bout_buff, int32(d1), int32(d2), convert(T, repulsion))
end