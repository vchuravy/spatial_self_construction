import OpenCL
const cl = OpenCL
import cl.Buffer, cl.CmdQueue, cl.Context, cl.Program

function getDiffusionKernel{T <: FloatingPoint}(::Type{T})
    nType = T == Float64 ? "double" : "float"

    return "
        #if defined(cl_khr_fp64)  // Khronos extension available?
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)  // AMD extension available?
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif

        #define Conc(x,y) a[y*D2 + x]
        #define Pot(x,y) b[y*D2 + x]

        #define P_move(x,y) out[y*D2 + x]

        __kernel void diffusion(
                      __global const $nType *a,
                      __global const $nType *b,
                      __global $nType *out,
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

        const $nType p = Pot(i, j);
        const $nType psw = Pot(south, west);
        const $nType psj = Pot(south, j   );
        const $nType pse = Pot(south, east);
        const $nType piw = Pot(i    , west);
        const $nType pie = Pot(i    , east);
        const $nType pnw = Pot(north, west);
        const $nType pnj = Pot(north, j   );
        const $nType pne = Pot(north, east);

        $nType ge[3][3];

        ge[0][0] = -1 * (psw - p) / ( 1 - exp(psw - p));
        ge[0][1] = -1 * (psj - p) / ( 1 - exp(psj - p));
        ge[0][2] = -1 * (pse - p) / ( 1 - exp(pse - p));
        ge[1][0] = -1 * (piw - p) / ( 1 - exp(piw - p));
        ge[1][1] = 0;
        ge[1][2] = -1 * (pie - p) / ( 1 - exp(pie - p));
        ge[2][0] = -1 * (pnw - p) / ( 1 - exp(pnw - p));
        ge[2][1] = -1 * (pnj - p) / ( 1 - exp(pnj - p));
        ge[2][2] = -1 * (pne - p) / ( 1 - exp(pne - p));

        $nType sumGE = 0;
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                if(isnan(ge[i][j])){
                    ge[i][j] = 1;
                }
                sumGE += ge[i][i];
            }
        }

        $nType concentration = Conc(i,j) / 9;

        //Inflow
        P_move(i,j) -= concentration * sumGE;

        //Outflow
        P_move(south,west) += ge[0][0] * concentration;
        P_move(south,j   ) += ge[0][1] * concentration;
        P_move(south,east) += ge[0][2] * concentration;
        P_move(i    ,west) += ge[1][0] * concentration;

        P_move(i    ,east) += ge[1][2] * concentration;
        P_move(north,west) += ge[2][0] * concentration;
        P_move(north,j   ) += ge[2][1] * concentration;
        P_move(north,east) += ge[2][2] * concentration;
    }
"
end

function diffusionCL!{T <: FloatingPoint}(
    a_buff :: Buffer{T}, b_buff :: Buffer{T},
    out_buff :: Buffer{T},
    d1 :: Int64, d2 :: Int64,
    ctx :: Context, queue :: CmdQueue, program :: Program)

    cl.copy!(queue, out_buff, zeros(T, d1, d2))

    k = cl.Kernel(program, "diffusion")

    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, out_buff, int32(d1), int32(d2))
end