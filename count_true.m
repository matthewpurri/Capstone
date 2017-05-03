function [true_pix] = count_true(Binary_Img)
i_r = size(Binary_Img,1);
i_c = size(Binary_Img,2);
true_pix = 0;
for r = 1:i_r
    for c = 1:i_c
        if(Binary_Img(r,c) == true) 
            true_pix = true_pix+1;
        end
    end
end

end