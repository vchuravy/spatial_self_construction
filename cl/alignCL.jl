import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getAlignKernel{T <: FloatingPoint}(:: Type{T})
        nType = T == Float64 ? "double" : "float"
        nPi = T == Float64 ? "M_PI" : "M_PI_F"

        return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif
        #define number $nType
        #define number8 $(nType)8
        #define PI $nPi

        #define Conc(x,y) a[y*D2 + x]
        #define Dir(x,y)  b[y*D2 + x]

        #define Newdir(x,y) out[y*D2 + x]

        static number sum(const number8 n) {
            return n.s0 + n.s1 +n.s2 + n.s3 + n.s4 + n.s5 + n.s6 + n.s7;
        }

        __kernel void align(
                      __global const number *a,
                      __global const number *b,
                      __global number *out,
                      const int D1,
                      const int D2,
                      const number attraction,
                      const number step) {

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

        // Getting the data and storing it in two eight vectors
        number direction = Dir(i, j);

        number8 conc;
        conc.s0 = Conc(south,west);
        conc.s1 = Conc(south,j   );
        conc.s2 = Conc(south,east);
        conc.s3 = Conc(i    ,east);
        conc.s4 = Conc(north,east);
        conc.s5 = Conc(north,j   );
        conc.s6 = Conc(north,west);
        conc.s7 = Conc(i    ,west);

        number8 dir;
        dir.s0 = Dir(south,west);
        dir.s1 = Dir(south,j   );
        dir.s2 = Dir(south,east);
        dir.s3 = Dir(i    ,east);
        dir.s4 = Dir(north,east);
        dir.s5 = Dir(north,j   );
        dir.s6 = Dir(north,west);
        dir.s7 = Dir(i    ,west);

        // Calculate
        number8 diff;

        diff = direction - dir;
        diff = fmod(diff, PI);
        diff = -sin(2 * diff);

        number dtheta = sum(conc * diff);

        dtheta = attraction * dtheta / 8;
        number ndir = direction + dtheta * step;

        Newdir(i,j) = fmod(ndir,PI);

    }
    "
end

function alignCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T},
    out_buff :: Buffer{T},
    attraction :: Real, step :: Real, d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    k = cl.Kernel(program, "align")

    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, out_buff, int32(d1), int32(d2), convert(T, attraction), convert(T, step))
end