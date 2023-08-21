# usb
There are 2 parts of code. 
1. generating the targeted UAP, in gene_uap. targeted_uap.py is an example of how to use it.

2. using USB for detection. Use run_detection.sh to call the methods.

Note, the generated UAP should be put at the same dir as the model dir. And the name of UAP file should be '/uap_tar_'+str(target)+'_mark('+mark_height+','+str(mark_width)+').pth'
