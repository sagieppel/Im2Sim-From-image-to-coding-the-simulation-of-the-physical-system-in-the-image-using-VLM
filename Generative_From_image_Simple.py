# This code uses OpenRouter API to generate a code that creates an image based on reference image
import cv2
import base64, requests
from pathlib import Path
import json
import os
API_key = "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"
InDir="/media/deadcrow/6TB/python_project/Im2Sim2Im_GIT/pixabay_downloads//"
OutDir="/media/deadcrow/6TB/python_project/Im2Sim2Im_GIT/Generate_pixabay//"
model="google/gemini-3-flash-preview"

##############################################################################################
def replicate_concept_simple(API_key,input_image_path,output_dir,model):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    output_img_path=output_dir+"/generated_image.jpg"

    prompt = ("Look at the image and write python code that recreates the content of the image as best as possible. "
              "The code should contain a function generate(out_path) that generated image and save it into out_path."
              "Do not display the image or use any GUI functions. Dont use matplotlib.pyplot."
              "Your response must come as a parsable json of the following format: {'code':<only code ready to execute>}"#,'describe':<describe what you see in the image>}."
              "Respond with raw JSON only. Do not use Markdown.Do not wrap the response in code fences. Output must be directly parsable by JSON.parse.")
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(Path(input_image_path).read_bytes()).decode() # encode input image as data URL
    content = [
                {"type": "text", "text": prompt},

                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
    r = requests.post( # send request to OpenRouter API
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_key}"},
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": content,
            }],
        },
    )
    txt=r.json()['choices'][0]["message"]['content'] # get the content of the response

    dic = json.loads(txt)  # parse the response as json
    code = dic['code']  # get the code
    if "matplotlib.pyplot." in code: return 0
    namespace1 = {}
    print("\n\n\n\n",code)
    exec(code,namespace1)

    namespace1["generate"](output_img_path)
    if os.path.exists(output_img_path):
        im=cv2.imread(output_img_path,0)
        if im.max()-im.min()>10:
            with open(output_dir + "//code.py", "w") as f: f.write(code)
            cv2.imwrite(output_dir + "//input_image.jpg",cv2.imread(input_image_path))
            with open(output_dir + "//finish.txt", "w") as f: f.write("Finished successfully")
            if 'describe' in dic:
                with open(output_dir + "//description.txt", "w") as f: f.write(dic['describe'])


##################################################################33333
if __name__ == "__main__":
    if not os.path.exists(OutDir): os.mkdir(OutDir)
    for fl in os.listdir(InDir):
        output_dir = os.path.join(OutDir,fl.replace(".jpg","").replace(".png","").replace(".",""))
        if os.path.exists(output_dir+"//finish.txt"): continue
        try:
           replicate_concept_simple(API_key,InDir+"//"+fl,output_dir,model)
        except Exception as error:
            debug_error=str(error)
            print("Error in processing file:",fl,"Error:",debug_error)