import OpenCL; const cl = OpenCL

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
            queue = CmdQueue(ctx)
            return (true, true, ctx, queue)
        elseif GPU_OCL
            println("Running simulation on GPU with OpenCL")
            devs = filter(has64Bit, cl.devices(:gpu))
            ctx = cl.Context(devs)
            queue = CmdQueue(ctx)
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
            if any(map(has64Bit, cl.devices(:gpu)))
                devs = filter(has64Bit, cl.devices(:gpu))
                ctx = cl.Context(devs)
                queue = CmdQueue(ctx)
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

