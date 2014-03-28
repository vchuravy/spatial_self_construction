include("main.jl")

function runProcess(config, val, out, loadTime)
    f, dist = val
    params = first(values(dist))
    r = try
        main(config, dist, true, loadTime = loadTime, resultFolder=out)
    catch
        (NaN, NaN, NaN, false, NaN, NaN, NaN, NaN, NaN, NaN, NaN, "")
    end
    return (f, params, r)
end