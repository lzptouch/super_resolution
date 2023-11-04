dataset_path='E:\dataset\SISR\Test\BSDS100_LR4\'
output_path='E:\dataset\SISR\Test\BSDS100_HR4_bic\'
    image_list = dir(fullfile(dataset_path));
    num_image = numel(image_list);
    for i=3:num_image
       image_name = image_list(i).name;
       image_ = im2double(imread([dataset_path,image_name]));
       image = imresize(image_,4,"bicubic");
       
       if ~exist("output_path", 'dir')
           mkdir(output_path);     
       end
       write_name = [output_path,image_name]
       imwrite(image, write_name);   
       if mod(i,100)==0
          fprintf('total: %d; output: %d; completed: %f%% \n',num_image, i, (i/num_image)*100) ;
       end
    end

 ;