import os
import shutil
import re
from tools import VisualQuestion as VQ
import json_pkl
import tools.MainFunctions as MF
import cv2
'''

Using LVLMs to extract simulation from an image of visual phenomena. 
This code use VLMS (GPT/Qwen/Gemini…) to look at images of visual patterns (clouds, waves) , identify the model of the physical system beyond the pattern, write it as code and run this code to generate a simulated image of the pattern.
It runs using the API of various VLMs (GP/GEMINI/QWEN/).
The code receive and folder of images (real_im_dir) and for each image generate and code model of the system formed the pattern in the image
Also it run this code to generate one simulated image of the the pattern
see __main__ section for parameters
Note you need to set API key in the API_KEY.py corresponding to the model you choose
'''

################################ Identify the model that form a pattern in image and implement in code ###########################################################################################

def pattern_from_image(texture_dir,out_dir,model="",display=False):
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

          #  # Infer  model/code  from image

          for kk in range(4):
              print("\n\n\nAttempt",kk,"\n\n\n"+out_path)
              txt = ("Look carefully at the pattern of the image, describe the pattern suggest what process created it."
                     " Then write python function simulate(im_path) that simulate this process, generate and save the resulting image and save it in im_path. Dont display the image on the screen!"
                     "Your response must come as json dictionary with the following fields:"
                     " {"
                     "'description': description of the pattern"
                     ",'process': description of the process/system/method that created this pattern"
                     ",'code': python function that simulate this process and generate the pattern as an image"
                     "}")
    #============================Generate code=========================================================================

              code_data = VQ.get_response_image_txt_json(text=txt, img_path=image_data, model=model)
    #==============================Run  and debug the code================================================================================
              task_description="Task: "+txt +"\nDescription:\n"+code_data['description']+"\nProcces:\n"+code_data['process']
              for ky in code_data:
                  with open(out_path+"//"+ky,"w") as fl: fl.write(code_data[ky])
              json_pkl.save_pkl(code_data,out_path+"//data.pkl")

              out_im_path =  out_path+"//image.jpg"
              code_path = out_path + "//generate.py"
# ------------Testing code usde to debug the generated code--------------------------------------
              testing_code_str = (
                        "\nimport importlib\n" +
                        "\nimport " + MF.path_to_import(code_path) + " as generate\n" +
                        # code_path.replace("//",".").replace(".py","").replace("/",".").replace("..",".") + " as generate\n"+
                        "\nimportlib.reload(generate)"
                        "\ngenerate.simulate('" + out_im_path + "')\n"
                )
#--------------Run debug the code --------------------------------------------------------

              code_verified, path, test_dir, code, captured_stdout, messages = (
                        MF.run_debug_code(messages=[], code=code_data['code'],
                                          code_dir=out_path, functions_and_var={}, codefilename="generate.py",
                                          testing_code=testing_code_str, clean_dir=False, task_description=task_description, time_out=9000, rechek_code=False, model=model))

#----------check for standart issues with output----------------------------------------------

              if not code_verified: continue
              if not os.path.exists(out_im_path): continue
              try:
                  im=cv2.imread(out_im_path)[0]
                  if im.max()-im.min()<18: continue # try to avoid uniform images
              except:
                   continue
              with open(out_path+"//success.txt","w") as fl: fl.write("succcess")
              break
#-------------------Finish and save results (for one image/mode)-----------------------------------

          with open(out_path + "//finish.txt", "w") as fl: fl.write("finish")
          if display:
                         cv2.imshow("Source:  " + imfl, cv2.imread(impath))
                         cv2.imshow("Result:  " + imfl, cv2.imread(out_im_path))
                         cv2.waitKey()




def run_multi_model(
        main_in_dir,
        main_outdir
): # Run the model generation  with mulitple models
    if not os.path.exists(main_outdir):
        os.mkdir(main_outdir)
#-------------Define models to use
    openai_models = ["gpt-5-mini", "gpt-5"]

    together_models = [#"google/gemma-3n-E4B-it",
                       "meta-llama/Llama-4-Scout-17B-16E-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]#,"deepseek-ai/DeepSeek-R1-0528-tput","deepseek-ai/DeepSeek-V3","moonshotai/Kimi-K2-Instruct"]
    gemini_models = ["gemini-2.5-pro", "gemini-2.5-flash"]
    grok_models = ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-4"]
    claude_models = ["claude-sonnet-4-5-20250929"]
    combine_list = openai_models + gemini_models + together_models + gemini_models + grok_models + claude_models


#------------Run each of the models in the list on the same folder



    for model in combine_list:
        model_simple_name = model.replace(".", "").replace(" ", "").replace("-", "_").replace("/", "_").replace(r"\\",r"_")
        for ii in range(10): print(main_outdir + "//" + model_simple_name)
        for i in range(10):
          try:
             pattern_from_image(texture_dir=main_in_dir, out_dir= main_outdir + "//" + model_simple_name, model=model, display=False)
             break
          except:
              continue

###############################################################################################3333333

# Main function

###################################################################################################
if __name__=="__main__":
         real_im_dir=r"sample_images//" #input folder with various of images of visual patters/phonomana that the LVLM will use to infer models from
         main_outdir = "Image2Sim2Im_output//" # Out dir where images and models will be saved (note this need to be subdir of) must be subfolder of the code folder and should be given in relative path. The reason is that some of the generated scripts created will be saved in this folder and will be imported and executed during the run.
         model = "gpt-5" # Model used for inference make sure you have the appropriate API key in the API_KEY.py file
         pattern_from_image(real_im_dir, main_outdir, model=model)

         # run_multi_model(
         #     main_in_dir=real_im_dir,
         #     main_outdir="..//Image2Sim2Im_all//"
         # )
         # pattern_from_image(texture_dir, out_dir="../texture_from_im_mini//", model="gpt-5-mini")