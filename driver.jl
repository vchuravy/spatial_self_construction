using MAT
using DataFrames

include("config.jl")

function getDisturbance(val)
    println(val)
    alpha, beta = val
    return {2.0 => [:punch_local, 20, 25, alpha, beta]}
end

function runCluster(min, max, steps, fileName)
    config = matread("data/$(fileName).mat")
    println("Loaded config")
    vals = linspace2d(min, max, steps)
    println("Calculated values to operate on")
    results = Dict()
    println("Starting computation now")
    for v in vals
        r = runProcess(config, v)
        results[v] = r
    end
    println("Create DF")
    df = resultsToDataFrame(results)
    show(df)
end

linspace2d(min, max, steps) = [(x,y) for x in linspace(min, max, steps), y in linspace(min, max, steps)]

function runProcess(config, v)
    println(v)
    dist = getDisturbance(v)
    println(dist)
    r = main(config, dist, true, loadTime = 1500)
    # r = (0,0)
    return r
end

function resultsToDataFrame(results)
    A = Array(Float64,0)
    B = Array(Float64,0)
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

    for (key, value) in results
        alpha, beta = key
        push!(A, alpha)
        push!(B, beta)
        t, ts, s, mm, ma, sm, sa, sf, sw, sd = value
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
    end
    data = DataFrame(alpha = A, beta = B, time = T, timeToStable = TS, stable = S, meanM = MM, meanA = MA, structM = SM, structA = SA, structF = SF, structW = SW, structD = SD)
    writetable("output.csv", data)
    return data
end


