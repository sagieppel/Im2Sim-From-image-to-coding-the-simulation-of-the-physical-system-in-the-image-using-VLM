import os
import shutil
import re
from tools import VisualQuestion as VQ
import json_pkl
import tools.MainFunctions as F2
import cv2

'''

Using LVLMs to extract simulation from an image of visual phenomena. 
This code use VLMS (GPT/Qwen/Gemini…) to look at images of visual patterns (clouds, waves) , identify the model of the physical system beyond the pattern, write it as code and run this code to generate a simulated image of the pattern.
It runs using the API of various VLMs (GP/GEMINI/QWEN/).
The code receive and folder of images (real_im_dir) and for each image generate and code model of the system formed the pattern in the image
Also it run this code to generate one simulated images of the the pattern
see __main__ section for parametre
Note you need to set API key in the API_KEY.py corresponding to the model you choose
'''
################################Run  image to image matching test ###########################################################################################

def pattern_from_image(texture_dir,out_dir,model="",num_images=10,display=False):
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    for imfl in os.listdir(texture_dir):
# -----------------Generate query------------------------------------------
          dname = "AA"+re.sub(r'[^A-Za-z0-9]', '', imfl[:-4])#}# '_',imfl[:-4])}"
          out_path=out_dir+"//"+dname +"//"
          if os.path.exists(out_path + "//finish.txt"): continue

          if not os.path.exists(out_path): os.mkdir(out_path)
          impath = texture_dir+"//"+imfl
          image_data = {"image": impath}
          shutil.copyfile(impath,out_path+"//"+imfl)
          cv2.imwrite(out_path+"//origin.jpg",cv2.imread(out_path+"//"+imfl))
          for kk in range(4):
              print("\n\n\nAttempt",kk,"\n\n\n"+out_path)
              txt = ("Look carefully at the pattern of the image, describe the pattern suggest what process that created it."
                     " Then write python function simulate(out_dir,num_images) that simulate this process, generate num_images images with the pattern  and save them in out_dir. Dont display the image on the screen!"
                     "Your response must come as json dictionary with the following fields:"
                     " {"
                     "'description': description of the pattern"
                     ",'process': description of the process/system/method/model that created this pattern"
                     ",'code': python function that simulate this process and generate the pattern as images (simulate(out_dir,num_images))"
                     "}")
    #============================Generate code=========================================================================
              #  # Infer  model/code  from image
              code_data = VQ.get_response_image_txt_json(text=txt, img_path=image_data, model=model)
    # ==============================Run  and debug the code================================================================================

              task_description="Task: "+txt +"\nDescription:\n"+code_data['description']+"\nProcces:\n"+code_data['process']
              for ky in code_data:
                  with open(out_path+"//"+ky,"w") as fl: fl.write(code_data[ky])
              json_pkl.save_pkl(code_data,out_path+"//data.pkl")

              out_im_path =  out_path+"//images//"
              code_path = out_path + "//generate.py"
              # ------------Testing code usde to debug the generated code--------------------------------------
              testing_code_str = (
                        "\nimport importlib\n" +
                        "\nimport " + F2.path_to_import(code_path) + " as generate\n" +
                        # code_path.replace("//",".").replace(".py","").replace("/",".").replace("..",".") + " as generate\n"+
                        "\nimportlib.reload(generate)"
                        "\ngenerate.simulate('"+out_im_path+"',"+str(num_images)+")\n"
                )
              # --------------Run debug the code --------------------------------------------------------
              code_verified, path, test_dir, code, captured_stdout, messages = (
                        F2.run_debug_code(messages=[], code=code_data['code'],
                                          code_dir=out_path, functions_and_var={}, codefilename="generate.py",
                                          testing_code=testing_code_str, clean_dir=False,task_description=task_description, time_out=12000, rechek_code=False,model=model))

              if not code_verified: continue # if code failed try again
              if not os.path.exists(out_im_path): continue

# ---------------------- check for issues if common issues found try again----------------------------------------------------------------

              fl_list = []
              for fl in os.listdir(out_im_path):
                  for tp in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".heic", ".heif"]:
                      if tp in fl:
                          fl_list.append(out_im_path + "//" + fl)
              if len(fl_list)<num_images-1: continue # not enough images generated
              try:
                  im0 = cv2.imread(fl_list[0], 0)
                  if im0.max() - im0.min() < 18: continue  # try to avoid uniform images

                  im1 = cv2.imread(fl_list[1], 0)
                  if im1.max() - im1.min() < 18: continue  # try to avoid uniform images

                  im1 = cv2.resize(im1, [im0.shape[1], im0.shape[0]])
                  if ((im1 - im0).__abs__() > 15).mean() < 0.01: continue  # try to avoid overly similar images

              except:
                  continue
              with open(out_path + "//success.txt", "w") as fl:
                  fl.write("succcess")
              break
#----------------------------------------------------------------------------------------


          with open(out_path + "//finish.txt", "w") as fl: fl.write("finish")
          if display:
                         cv2.imshow("Source:  " + imfl, cv2.imread(impath))
                         cv2.imshow("Result:  " + imfl, cv2.imread(out_im_path))
                         cv2.waitKey()





###############################################################################################3333333

# Main function

###################################################################################################
if __name__=="__main__":
    real_im_dir = r"sample_images//"  # input folder with various of images of visual patters/phonomana that the LVLM will use to infer models from
    main_outdir = "Image2Sim2Im_output_multi//"  # Out dir where images and models will be saved (note this need to be subdir of) must be subfolder of the code folder and should be given in relative path. The reason is that some of the generated scripts created will be saved in this folder and will be imported and executed during the run.
    model = "gpt-5"  # Model used for inference make sure you have the appropriate API key in the API_KEY.py file
    num_images=10 # number of images to generate for each model
    pattern_from_image(texture_dir = real_im_dir, out_dir=main_outdir,num_images=num_images, model=model)