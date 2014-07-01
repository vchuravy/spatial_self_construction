function area{T <: Real}(direction :: Matrix{T}, LONG :: Real)
    d1, d2 = size(direction)
    out = similar(direction, (4,d1,d2))

    area!(out, direction, LONG)
    return(out)
end

function area!{T <: Real}(out :: Array{T, 3}, direction :: Matrix{T}, LONG :: Real)
    D1, D2 = size(direction)
    SHORT = one(T)
    PI_4 = pi/4
    PI = pi

    for j in 1:D2
        for i in 1:D1
                dir = direction[i,j];

                area1 = LONG*SHORT/2 * (   PI_4/2-dir - atan((SHORT-LONG)*sin(2*(  PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(  PI_4/2-dir)))));
                area2 = LONG*SHORT/2 * ( 3*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(3*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(3*PI_4/2-dir)))));
                area3 = LONG*SHORT/2 * ( 5*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(5*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(5*PI_4/2-dir)))));
                area4 = LONG*SHORT/2 * ( 7*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(7*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(7*PI_4/2-dir)))));
                area5 = LONG*SHORT/2 * ( 9*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(9*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(9*PI_4/2-dir)))));

                lps = SHORT * LONG * PI

                out[1, i, j] = (area2-area1)/lps
                out[2, i, j] = (area3-area2)/lps
                out[3, i, j] = (area4-area3)/lps
                out[4, i, j] = (area5-area4)/lps
        end
end