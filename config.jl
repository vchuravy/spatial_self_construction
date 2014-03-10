###
# Configuration
###
include("drawcircle.jl")

function loadConfig(dict)
    for key in keys(dict)
        sym = symbol(key)
        @eval $sym = $(dict[key])
    end
end

function loadVarConfig(keys, dict)
    for key in keys
        @assert key in keys(dict)
        sym = symbol(key)
        @eval $sym = $(dict[key])
    end
end

function saveConfig(keys)
    dict = Dict{String, Any}()
    for key in keys
        sym = symbol(key)
        @eval $dict[$key] = $sym
    end
    return dict
end

baseConfig = {
    "fieldSize"             => 2400,
    "fieldRes"              => 40,
    "timeTotal"             => 5000,
    "stepIntegration"       => 0.5,
    "visInterval"           => 5,
    "stepVisualization"     => 10,
    "epsilon"               => 0.001,
    "stableTime"            => 5000,
    "v_mR"                  => 180, # membraneRadius
    "v_mT"                  => 170,  # membraneThickness
    "v_fD"                   => 20, # food refill distance
    "avgM"                  => 0.8, #initial concentration of M in membrane
    "avgA"                  => 0.6,
    "avgF"                  => 0.18,
    "avgT"                  => 0.0,
    "diffusionF"            => 20, # absolute diffusion rate
    "diffusionA"            => 20, # absolute diffusion rate
    "diffusionM"            => 20, # absolute diffusion rate
    "diffusionW"            => 20, # absolute diffusion rate
    "attractionRate"        => 0.1, # attraction of directionality

    "MW_repulsion"          => 7.0,
    "MA_repulsion"          => 7.0,
    "MM_repulsion"          => 0.5,
    "AA_repulsion"          => 0.5,
    "long_direction"        => 20.0,

    "decayA"                => 0.001, # this is per timestep, not integration step.
    "decayM"                => 0.001,

    # rules for food
    "flowRateF"             => 0.8,

    # reaction 1 : a11*A + f11*F + m11*M + w11*W <-> a12*A + f12*F + m12*M + w12*W
    "a11" => 2, "f11" => 1, "m11" => 0, "w11" => 0, # negative numbers inhibit
    "a12" => 3, "f12" => 0, "m12" => 0, "w12" => 0, # inhibition needs to balanced for there not to be creation

    "kf1" => 0.1, # forward reaction rate
    "kb1" => 0.0, # backwards reaction rate

    # reaction 2 : a21*A + f21*F + m21*M + w21*W <-> a22*A + f22*F + m22*M + w22*W
    "a21" => 2, "f21" => 1, "m21" => 0, "w21" => 0,
    "a22" => 2, "f22" => 0, "m22" => 1, "w22" => 0,

    "kf2" => 0.1, # forward reaction rate
    "kb2" => 0.0, # backwards reaction rate

    #reaction 3 : a31*A + f31*F + m31*M + w31*W <-> a32*A + f32*F + m32*M + w32*W
    "a31" => 0, "f31" => 0, "m31" => 0, "w31" => 0,
    "a32" => 0, "f32" => 0, "m32" => 0, "w32" => 0,

    "kf3" => 0.0, # forward reaction rate
    "kb3" => 0.0 # backwards reaction rate
}

dataVars = ["history_W", "history_A", "history_M", "history_M_pot", "history_F", "history_dir", "Avec", "Fvec", "Mvec", "Wvec", "DAvec", "DMvec"]

function createStorageVars()
    @eval begin
        history_A = nothing
        history_F = nothing
        history_M = nothing
        history_M_pot = nothing
        history_W = nothing
        history_dir = nothing

        Avec = nothing
        Fvec = nothing
        Tvec = nothing
        Mvec = nothing
        Wvec = nothing
        DAvec = nothing
        DMvec = nothing
    end
end

function updateDependentValues()
    @eval begin
        saturationF = avgF
        fieldSizeY = fieldSize
        fieldSizeX = fieldSizeY
        fieldResY = fieldRes
        fieldResX = fieldResY
        _xc = fieldSize/2 # circle center's x position
        _yc = fieldSize/2 # circle center's y position


        # adjust membrane for given resolution
        xc = ceil(_xc*fieldRes/fieldSize)
        yc = ceil(_yc*fieldRes/fieldSize)
        mR = ceil(v_mR*fieldRes/fieldSize) # membraneRadius
        mT = ceil(v_mT*fieldRes/fieldSize) # membraneThickness
        fD = ceil(v_fD*fieldRes/fieldSize) # membraneThickness

        Frefill = zeros(fieldRes, fieldRes)
        for r in 0:0.01:(mR+mT+fD)
            drawcircle!(Frefill, xc, yc, r)
        end
        FrefillBinMask = Frefill .< 0.5
    end
end

