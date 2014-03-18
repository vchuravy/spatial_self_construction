using MAT
using DataFrames
using Datetime
using JSON

include("config.jl")
include("utils.jl")

jobs = Dict[]
working_on = Dict{Int, RemoteRef}()
idle = Int[]

function runCluster(fileName, outFolder = "results"; configName = "cluster_configs/all.json", loadTime=-1)
    try
        mkdir(outFolder)
    end

    dt = now()
    out = "$outFolder/$(year(dt))-$(month(dt))-$(day(dt))_$(hour(dt)):$(minute(dt)):$(second(dt))"
    mkdir(out)

    config = matread("data/$(fileName).mat")

    results = Any[]

    jobs = parseClusterConfig(configName)

    # Setup everything
    nc, ng, c_ocl = determineWorkload()

    addprocs(nc + ng)
    idle = workers()
    require("job.jl")

    count = 1
    for id in workers()
        if count <= nc
            if c_ocl
                @at_proc id CPU_OCL = true
            else
                @at_proc id CPU_OCL = false
            end
            @at_proc id GPU_OCL = false
        else
            @at_proc id CPU_OCL = false
            @at_proc id GPU_OCL = true
        end
        count += 1
    end

    println("Starting simulations on $(nworkers()) workers")

    try
        running = true
        while running
            for (w, rref) in working_on
                if isready(rref)
                    rref = pop!(working_on, w) # remove from dict
                    result = fetch(rref) # fetch Remote Reference
                    push!(results, result) # store the result
                    push!(idle, w) # mark worker as idle
                end
            end

            while (length(idle) > 0) && (length(jobs) > 0)
                work_item = pop!(jobs) # get a work item
                w = pop!(idle) # get an idle worker
                rref = @spawnat w runProcess(config, work_item, out, loadTime)
                working_on[w] = rref
            end

            println("Running jobs: $(length(working_on))")
            println("Jobs left: $(length(jobs))")

            if (length(jobs) == 0) && (length(working_on) == 0)
                running = false
            else
               timedwait(anyready, 120.0, pollint=0.5)
            end
        end
    catch e
        println(e)
    finally
        df = resultsToDataFrame(results, out)
        show(df)
    end
end

function anyready()
    for rref in values(working_on)
        if isready(rref)
            return true
        end
    end
    return isempty(working_on)
end

function parseClusterConfig(filePath)
    config = JSON.parse(open(filePath, "r"))
    vals = Dict[]

    for (name,  subconfig) in config
        subvals =
            if name == "punch_local"
                parsePunchLocal(subconfig)
            elseif name == "punch_random"
                parsePunchRandom(subconfig)
            elseif name == "gaussian_blur"
                parseGaussianBlur(subconfig)
            elseif name == "global"
                parseGlobal(subconfig)
            else
                warn("Can't handle $name")
                Dict[]
            end
        append!(vals, subvals)
    end
    return vals
end

function parsePunchLocal(config)
    x = config["x"]
    y = config["y"]
    alpha = config["alpha"]
    beta = config["beta"]

    ranges = map(torange, (x, y, alpha, beta))
    createDisturbance(:punch_local, ranges)
end

function parsePunchRandom(config)
    times = config["times"]
    alpha = config["alpha"]
    beta = config["beta"]
    ranges = map(torange, (alpha, beta))

    vals = Dict[]
    dist = createDisturbance(:punch_random, ranges)
    for i in 1:times
        append!(vals, dist)
    end
    return vals
end

function parseGaussianBlur(config)
    sigma = config["sigma"]
    createDisturbance1(:gaussian_blur, torange(sigma))
end

function parseGlobal(config)
    mu = config["mu"]
    sig = config["sig"]

    ranges = map(torange, (mu, sig))
    createDisturbance(:global, ranges)
end

function resultsToDataFrame(results, folder)
    Name = Array(Any,0)
    Param = Array(Any,0)
    T = Array(Float64,0)
    TS = Array(Float64,0)
    S = Array(Bool,0)
    MM = Array(Float64,0)
    MA = Array(Float64,0)
    SM = Array(Float64,0)
    SA = Array(Float64,0)
    SF = Array(Float64,0)
    SW = Array(Float64,0)
    SD = Array(Float64,0)
    FN = Array(Any,0)

    for ((name, p), value) in results
        t, ts, s, mm, ma, sm, sa, sf, sw, sd, fn = value
        push!(Name, name)
        push!(Param, p)
        push!(T, t)
        push!(TS, ts)
        push!(S, s)
        push!(MM, mm)
        push!(MA, ma)
        push!(SM, sm)
        push!(SA, sa)
        push!(SF, sf)
        push!(SW, sw)
        push!(SD, sd)
        push!(FN, fn)
    end
    data = DataFrame(name=Name, parameter=Param, time = T, timeToStable = TS, stable = S, meanM = MM, meanA = MA, structM = SM, structA = SA, structF = SF, structW = SW, structD = SD, fileName = FN)
    writetable("$folder/data.csv", data)
    return data
end


