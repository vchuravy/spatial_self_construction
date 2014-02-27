function drawcircle!(mat :: Matrix, xc, yc, r)

    x = 0;
    y = r;
    p = 1 - r;

     while x < y

         mat[round(xc+x), round(yc+y)] = 1
         mat[round(xc-x), round(yc+y)] = 1
         mat[round(xc+x), round(yc-y)] = 1
         mat[round(xc-x), round(yc-y)] = 1
         mat[round(xc+y), round(yc+x)] = 1
         mat[round(xc-y), round(yc+x)] = 1
         mat[round(xc+y), round(yc-x)] = 1
         mat[round(xc-y), round(yc-x)] = 1

         x += 1

         if p < 0
            p = p + 2 * x + 1
         else
            y = y - 1
            p = p + 2 * (x - y) + 1
         end

         mat[round(xc+x), round(yc+y)] = 1
         mat[round(xc-x), round(yc+y)] = 1
         mat[round(xc+x), round(yc-y)] = 1
         mat[round(xc-x), round(yc-y)] = 1
         mat[round(xc+y), round(yc+x)] = 1
         mat[round(xc-y), round(yc+x)] = 1
         mat[round(xc+y), round(yc-x)] = 1
         mat[round(xc-y), round(yc-x)] = 1
     end
end