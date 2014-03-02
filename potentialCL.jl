import OpenCL
const cl = OpenCL

function getPotentialKernel(repulsion :: Float32, long :: Float32, d1 :: Int, d2 :: Int)
    return "

    #define D1 $d1
    #define D2 $d2
    #define selfRepulsion $(repulsion)f
    #define LONG $(long)f
    #define SHORT 1.0f

    #define Afield(x,y) a[x*$d1 + y]
    #define Bfield(x,y) b[x*$d1 + y]
    #define Direction(x,y) d[x*$d1 + y]

    #define Apot(x,y) aout[x*$d1 + y]
    #define Bpot(x,y) bout[x*$d1 + y]

    float area(const float x, const float d) {
        const float p8 = x * M_PI/8;
        return LONG * SHORT / 2 * ( p8 - d - atan((SHORT - LONG)*sin(2*(p8 - d)) / (SHORT + LONG + (SHORT - LONG)*cos(2*(p8 - d)))));
    }

    __kernel void potential(
                      __global const float *a,
                      __global const float *b,
                      __global const float *d,
                      __global float *aout,
                      __global float *bout) {

        int j = get_global_id(0);
        int i = get_global_id(1);
        int west = j-1;
        int east = j+1;
        int north = i+1;
        int south = i-1;

        if(j == 1)
            west = D2;
        if(j == D2)
            east = 1;
        if(i == D1)
            north = 1;
        if(i == 1)
            south = D1;

        const float dir = Direction(i,j);
        const float bF = Bfield(i,j);
        const float cRepulsion = selfRepulsion * bF;

        Apot(i,j) += cRepulsion;

        const float area1 = area(1.0f, dir);
        const float area2 = area(3.0f, dir);
        const float area3 = area(5.0f, dir);
        const float area4 = area(7.0f, dir);
        const float area5 = area(9.0f, dir);

        const float lps = SHORT * LONG * M_PI;

        const float a1 = (area2-area1)/lps;
        const float a2 = (area3-area2)/lps;
        const float a3 = (area4-area3)/lps;
        const float a4 = (area5-area4)/lps;

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

function potentialCL(Afield :: Array{Float32, 2}, Bfield :: Array{Float32, 2}, direction :: Array{Float32, 2}, repulsion :: Real, long :: Real)
    d1, d2 = size(Afield)

    Apotential = zeros(Float32, d1, d2)
    Bpotential = zeros(Float32, d1, d2)

    device, ctx, queue = cl.create_compute_context()

    a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=Afield)
    b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=Bfield)
    d_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=direction)
    aout_buff = cl.Buffer(Float32, ctx, :w, d1 * d2)
    bout_buff = cl.Buffer(Float32, ctx, :w, d1 * d2)

    const clSource = getPotentialKernel(float32(repulsion), float32(long),d1, d2)

    p = cl.Program(ctx, source=clSource) |> cl.build!
    k = cl.Kernel(p, "potential")

    cl.call(queue, k, (d1,d2), nothing, a_buff, b_buff, d_buff, aout_buff, bout_buff)

    cl.copy!(queue, Apotential, aout_buff)
    cl.copy!(queue, Bpotential, bout_buff)

    return (convert(Array{Float64}, Apotential), convert(Array{Float64}, Bpotential))
end