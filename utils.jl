import OpenCL; const cl = OpenCL
using Cartesian

macro at_proc(p, ex)
   quote
           remotecall( $p, ()->eval(Main,$(Expr(:quote,ex))))
   end
end

function has64Bit(device)
    amd = "cl_amd_fp64"
    khr = "cl_khr_fp64"
    ext = cl.info(device, :extensions)

    return (khr in ext) || (amd in ext)
end

function countOCL()
    cpu_count = 0
    gpu_count = 0
    for d in cl.devices(:cpu)
        if has64Bit(d)
            cpu_count = 1 # We currently only support one device of each sort.
        end
    end
    for d in cl.devices(:gpu)
        if has64Bit(d)
            gpu_count = 1 # We currently only support one device of each sort.
        end
    end

    return {:gpu => gpu_count, :cpu => cpu_count}
end


function determineWorkload()
    cpus = CPU_CORES
    count = countOCL()

    CPU_OCL = count[:cpu] >= 1
    ncpu_workers =  if CPU_OCL
        iceil((CPU_CORES - count[:gpu] - 1) / 3) # If we have opencl support on the cpu we really don't want to spawn more then one process per 3 cores.
    else
        (CPU_CORES - count[:gpu] - 1) # Native Julia workers
    end

    ngpu_workers = count[:gpu]

    return (ncpu_workers, ngpu_workers, CPU_OCL)
end


function determineCapabilities(cluster :: Bool = false, allow32Bit = false, forceJuliaImpl = false)
    if cluster
        if CPU_OCL
            println("Running simulation on CPU with OpenCL")
            devs = filter(has64Bit, cl.devices(:cpu))
            ctx = cl.Context(devs)
            queue = cl.CmdQueue(ctx)
            return (true, true, ctx, queue)
        elseif GPU_OCL
            println("Running simulation on GPU with OpenCL")
            devs = filter(has64Bit, cl.devices(:gpu))
            ctx = cl.Context(devs)
            queue = cl.CmdQueue(ctx)
            return (true, true, ctx, queue)
        else
            println("Running simulation on CPU")
            return (true, false, nothing, nothing)
        end
    else
        try
            if forceJuliaImpl
                error("forced usage of Julia implementation.")
            end
            if any(map(has64Bit, cl.devices()))
                devs = filter(has64Bit, cl.devices())
                ctx = cl.Context([first(devs)])
                queue = cl.CmdQueue(ctx)
                return (true, true, ctx, queue)
            else
                warn("No OpenCL device with Float64 support found!")
                if allow32Bit
                    warn("Searching for device with Float32 support.")
                    device, ctx, queue = cl.create_compute_context()
                    return (false, true, ctx, queue)
                else
                    throw(Exception())
                end
            end
        catch e
            println("Got exception: $e")
            warn("OpenCL is not supported falling back to Julia computation")
            return (true, false, nothing, nothing)
        end
    end
end

# TODO: find a better way to handle arbitrary dimensions.
function createDisturbance(name :: Symbol, ranges)
    dim = length(ranges)
    if dim == 1
        createDisturbance1(name, ranges)
    elseif dim == 2
        createDisturbance2(name, ranges)
    elseif dim == 3
        createDisturbance3(name, ranges)
    elseif dim == 4
        createDisturbance4(name, ranges)
    else
        error("Can't handle dim $dim")
    end
end

function createDisturbance4(name :: Symbol, ranges)
    vals = Dict[]

    @nloops 4 i d -> ranges[d] begin
        args = @ntuple 4 i
        push!(vals, {100.0 => [name, args...]})
    end

    return vals
end

function createDisturbance3(name :: Symbol, ranges)
    vals = Dict[]

    @nloops 3 i d -> ranges[d] begin
        args = @ntuple 3 i
        push!(vals, {100.0 => [name, args...]})
    end

    return vals
end

function createDisturbance2(name :: Symbol, ranges)
    vals = Dict[]

    @nloops 2 i d -> ranges[d] begin
        args = @ntuple 2 i
        push!(vals, {100.0 => [name, args...]})
    end

    return vals
end

function createDisturbance1(name :: Symbol, range)
    vals = Dict[]

    for i in range
        push!(vals, {100.0 => [name, i]})
    end

    return vals
end

function torange(c :: Dict)
    min = c["min"]
    max = c["max"]
    step = c["step"]
    return min:step:max
end

function torange(c :: Real)
    return c
end

function writedlm_row(io::IO, row, dlm = ',')
    pb = PipeBuffer()
    state = start(row)
    while !done(row, state)
        (x, state) = next(row, state)
        Base.writedlm_cell(pb, x, dlm)
        done(row, state) ? write(pb,'\n') : print(pb,dlm)
    end
    (nb_available(pb) > (16*1024)) && write(io, takebuf_array(pb))
    write(io, takebuf_array(pb))
    nothing
end

function apply_punch_down!(A, x0, y0, a, b)
                d1, d2 = size(A)
                @assert x0 in 1:d1
                @assert y0 in 1:d2
                #dist(x, y) = sqrt((x - x0)^2 + (y-y0)^2)
                #dist(x, y) = abs(x-x0) + abs(y-y0)
                dist(x, y) = floor(sqrt((x - x0)^2 + (y-y0)^2))
                #punch(d) = 1- sech(1/b * d) ^ a
                punch(x)=a*exp(-x^2/(2*b^2))+1
                for j in 1:d2
                    for i in 1:d1
                        d = dist(i, j)
                        A[i,j] *= punch(d)
                    end
                end
            end