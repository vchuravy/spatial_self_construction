function areaJl!{T <: FloatingPoint}(direction :: Array{T, 2}, dout :: Array{Number4{T}, 2},  LONG :: Real)
    D1, D2 = size(direction)
    SHORT = one(T)
    PI_4 = pi/4
    PI = pi

    for i in 1:length(direction)
            dir = direction[i];

            area1 = LONG*SHORT/2 * (   PI_4/2-dir - atan((SHORT-LONG)*sin(2*(  PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(  PI_4/2-dir)))));
            area2 = LONG*SHORT/2 * ( 3*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(3*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(3*PI_4/2-dir)))));
            area3 = LONG*SHORT/2 * ( 5*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(5*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(5*PI_4/2-dir)))));
            area4 = LONG*SHORT/2 * ( 7*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(7*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(7*PI_4/2-dir)))));
            area5 = LONG*SHORT/2 * ( 9*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(9*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(9*PI_4/2-dir)))));

            lps = SHORT * LONG * PI;

            s0 = (area2-area1)/lps;
            s1 = (area3-area2)/lps;
            s2 = (area4-area3)/lps;
            s3 = (area5-area4)/lps;

            dout[i] = Number4(s0, s1, s2, s3);
    end
    return dout
end