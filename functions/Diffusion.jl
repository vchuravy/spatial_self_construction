function diffusion{T <: FloatingPoint}(concentration :: Matrix{T}, flow :: Matrix{Matrix{T}}, diff :: Real)
    p_move = similar(concentration)

    diffusion!(p_move, concentration, flow, diff)

    return p_move
end


function diffusion!{T <: FloatingPoint}(p_move :: Matrix{T}, concentration :: Matrix{T}, flow :: Matrix{Matrix{T}}, diff :: Real) #concentration and direction
    d1, d2 = size(concentration)

    conc_neighbourhood = similar(concentration, (3,3))
    flow_neighbourhood = similar(flow, (3,3))
    local_flow = similar(concentration, (3,3))

    for j in 1:d2
        for i in 1:d1

            get_moore!(conc_neighbourhood, concentration, i, j, d1, d2)
            get_moore!(flow_neighbourhood, flow, i, j, d1, d2)

            ###
            # Get the flow out of cell ij
            ###

            outflow =  centre(conc_neighbourhood) * sum(flow[i,j])

            ###
            # Collect the outflow from other cells
            ###

            translated_copy!(local_flow, flow_neighbourhood)

            ###
            # Calculate the inflow based on the outflow from other cells into this on.
            ###

            inflow = 0

            zero_centre!(conc_neighbourhood) # Exclusive moore-neighbourhood

            @simd for c in 1:9
                @inbounds inflow += conc_neighbourhood[c] * local_flow[c]
            end

            ###
            # Inflow - outflow = change
            ###

            p_move[i,j] = diff * (inflow - outflow)
        end
    end
end