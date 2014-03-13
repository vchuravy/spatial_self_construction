using MAT
using DataFrames

include("config.jl")

function getDisturbance(val)
    alpha, beta = val
    return {2.0 => [:punch_local, 20, 25, alpha, beta]}
end

function runCluster(min, max, steps, fileName, out = "")
    config = matread("data/$(fileName).mat")
    m1, m2 = max
    mi1, mi2 = min
    s1, s2 =steps
    vals = linspace2d(m1, mi1, s1, m2, mi2, s2)
    results = Dict()
    for v in vals
        r = runProcess(config, v)
        results[v] = r
    end
    df = resultsToDataFrame(results)
    show(df)
end

linspace2d(min1, max1, steps1, min2, max2, steps2) = [(x,y) for x in linspace(min1, max1, steps1), y in linspace(min2, max2, steps2)]

function runProcess(config, v)
    dist = getDisturbance(v)
    r = main(config, dist, true, loadTime = 1500)
    println("$v, $r")
    return r
end

function resultsToDataFrame(results, fileName)
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
    writetable("$fileName.csv", data)
    return data
end


