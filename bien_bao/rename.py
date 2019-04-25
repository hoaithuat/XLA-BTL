import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir("cam_do_xe"): 
        dst ="cam_do_xe_" + str(i) + ".png"
        src ='cam_do_xe/'+ filename 
        dst ='cam_do_xe/'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 