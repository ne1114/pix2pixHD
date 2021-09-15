################################ Testing ################################
# labels only
# python test.py --name 0908Slice_wo_model --netG local --ngf 32 --dataroot datasets/preprocessing-d20210908_vertical_slice/ --input_nc 3 --output_nc 3 --label_nc 0 --no_instance
python test.py --name 0910Slice_3slices_global_resizeNcrop --dataroot datasets/preprocessing-d20210910_vertical_slice/ --input_nc 3 --output_nc 3 --label_nc 0 --no_instance