import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getDiffusionKernel{T <: FloatingPoint}(:: Type{T})
        nType = T == Float64 ? "double" : "float"
        nOne = T == Float64 ? "1.0" : "1.0f"

        return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif
        #define number $nType
        #define number8 $(nType)8

        #define Conc(x,y) a[y*D2 + x]
        #define Pot(x,y) b[y*D2 + x]
        #define Flow(x,y) f[y*D2 + x]

        #define P_move(x,y) out[y*D2 + x]

        __kernel void flow(
                      __global const number *b,
                      __global number8 *f,
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

            const number p = Pot(i,j);

            number8 ge;
            ge.s0 = Pot(south,west);
            ge.s1 = Pot(south,j   );
            ge.s2 = Pot(south,east);
            ge.s3 = Pot(i    ,east);
            ge.s4 = Pot(north,east);
            ge.s5 = Pot(north,j   );
            ge.s6 = Pot(north,west);
            ge.s7 = Pot(i    ,west);

            ge = ge - p;

            ge = -1 * ge / (1-exp(ge));

            //Check for NaN
            const number8 ones = (number8)($nOne);
            ge = select(ge, ones, isnan(ge));
            Flow(i,j) = ge / 8;

        }

        static number sum(const number8 n) {
            return n.s0 + n.s1 +n.s2 + n.s3 + n.s4 + n.s5 + n.s6 + n.s7;
        }

        __kernel void diffusion(
                      __global const number *a,
                      __global const number8 *f,
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

            // Get the flow out of cell ij

            number outflow =  Conc(i,j) * sum(Flow(i,j));

            // Calculate the inflow based on the outflow from other cells into this on.

            number inflow  =    Conc(south, west) * Flow(south, west).s4 +
                                Conc(south, j   ) * Flow(south, j   ).s5 +
                                Conc(south, east) * Flow(south, east).s6 +
                                Conc(i    , east) * Flow(i    , east).s7 +
                                Conc(north, east) * Flow(north, east).s0 +
                                Conc(north, j   ) * Flow(north, j   ).s1 +
                                Conc(north, west) * Flow(north, west).s2 +
                                Conc(i    , west) * Flow(i    , west).s3 ;

            // Inflow - outflow = change

            P_move(i,j) = inflow - outflow;
    }
"
end

function diffusionCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T},
    out_buff :: Buffer{T},
    d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    flow_buff = cl.Buffer(T, ctx, :rw, d1 * d2 * 8)

    kd = cl.Kernel(program, "diffusion")
    kf = cl.Kernel(program, "flow")

    cl.call(queue, kf, (d1,d2), nothing, b_buff, flow_buff, int32(d1), int32(d2))
    cl.call(queue, kd, (d1,d2), nothing, a_buff, flow_buff, out_buff, int32(d1), int32(d2))
end