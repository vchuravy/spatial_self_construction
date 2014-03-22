using MAT
using DataFrames
using Datetime
using JSON

include("config.jl")
include("utils.jl")

jobs = Dict[]
resultStreams = Dict()
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

    jobs, resultStreams = parseClusterConfig(configName, out)

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
                    fN, params, result = fetch(rref) # fetch Remote Reference
                    writedlm_row(resultStreams[fN], [params, result...])
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
		sleep(30.0)
	    	# timedwait(1.0, pollint=0.5) do
		#	rrefs = values(working_on)
		#	isempty(rrefs) || any(isready, rrefs)
		# end
            end
        end
    catch e
        println(e)
    finally
        map(close, values(resultStreams))
    end
end

function parseClusterConfig(filePath, outputPath)
    config = JSON.parse(open(filePath, "r"))
    vals = (String, Dict)[]
    outputStreams = Dict()

    for (fileName,  subconfig) in config
        if haskey(subconfig, "name")
            name = subconfig["name"]
            parameterNames, subvals =
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
            times = if haskey(subconfig, "times")
                        subconfig["times"]
                    else
                        1
                    end
            subvals = [(fileName, v) for v in subvals]
            for i in 1:times
                append!(vals, subvals)
            end
            out = open("$outputPath/$fileName", "w")
            writedlm_row(out, ["name", parameterNames, "time", "timeToStable", "stable", "meanMField", "meanAField", "structM", "structA", "structF", "structW", "structD", "fileName"])
            outputStreams[fileName] = out
        else
            warn("$fileName has no disturbance assign to it.")
        end
    end
    return (vals, outputStreams)
end

function parsePunchLocal(config)
    x = config["x"]
    y = config["y"]
    alpha = config["alpha"]
    beta = config["beta"]

    ranges = map(torange, (x, y, alpha, beta))
    (["x", "y", "alpha", "beta"], createDisturbance(:punch_local, ranges))
end

function parsePunchRandom(config)
    alpha = config["alpha"]
    beta = config["beta"]
    ranges = map(torange, (alpha, beta))

    (["alpha", "beta"], createDisturbance(:punch_random, ranges))
end

function parseGaussianBlur(config)
    sigma = config["sigma"]
    (["sigma"], createDisturbance1(:gaussian_blur, torange(sigma)))
end

function parseGlobal(config)
    mu = config["mu"]
    sig = config["sig"]

    ranges = map(torange, (mu, sig))
    (["mu", "sig"], createDisturbance(:global, ranges))
end