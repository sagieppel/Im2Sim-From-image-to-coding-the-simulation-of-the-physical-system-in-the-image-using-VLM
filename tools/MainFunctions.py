import shutil
import tools.Code_Exec as Code_Exec
import textwrap
import tools.VisualQuestion as VQ
import os


import importlib
import re
import json

###############################################################################
def sanitize_code(code_str: str) -> str:
    """
    Clean up common Unicode punctuation.
    """
    for old, new in [('“','"'),('”','"'),('’','\''),('‘','\''),('—','-'),('–','-')]:
        code_str = code_str.replace(old, new)
    return code_str.strip()
###############################################################################################################################

# Check code and install whatever dependencies needed to  run the code

###############################################################################################################
def check_and_install_dependencies(code,model="o4-mini",num_tries=3,messages=None):
    if messages is None:
        prompt = ("Read the following code and see which packages/imports/dependencies does it use.\n"
                "Write  python script that check if all packages/imports/dependencies are available and install them if necessary\n"
                "\nThe answer most come in json format of {'packages': list of packages you need to install or 'None' if there arent any', 'installation_code': python code that check if the packages installed and install them if necessary (the code most be ready to run with no extra text). If no installations are needed leave empty.")
        messages = [
                     {"role": "system", "content": "You are a software developer ."},
                     {"role": "user", "content": prompt},
                     {"role": "user", "content": "Here is the code: \n\n"+code}

                   ]
    print(messages)



    for i in range(num_tries):
        results = VQ.get_reponse(messages=messages,model=model,as_json=True)
        messages.append({"role": "system", "content": str(results)})
        print(messages[-1])
        # try:
        #     results = json.loads(MainFunctions.normalize_to_json(raw))
        # except:
        #     messages.append({"role": "user", "content":"Couldnt parse your response to json using json.load, please try again"})
        #     print(messages[-1])
        #     continue
        if i==0:
            if results['packages'] == None or results['packages'] == [] or results['packages'] == "none" or len(
                results['installation_code']) == 0: return True, messages, ""
            code_to_run = results['installation_code']
        else:
            if "yes" in results['solvable'].lower():
                code_to_run = results['fixed_code']
            else:
                return False, messages, ""


        print("Trying to install dependencies using:\n\n", textwrap.dedent(code_to_run))
        successed, captured_stdout, captured_stderr = Code_Exec.run_code(textwrap.dedent(code_to_run))
        if successed:   return True, messages, code_to_run

        messages.append({"role": "user", "content": "Failed installation with error:\n"+captured_stderr+" \n\n Try to solve the error. \n Give me your output as json file in the format:"
                       + " {'packages': list of packages you need to install or 'None' if there arent any\n,'solvable':can you solve the issue single word  answer: yes/no\n,'fixed_code':The fixed code ready to run}"})
        print(messages[-2:])
#--------------------------------------------------------------------------


    return False,messages, ""
##############################################################################################################33

# run code inspect results and debug

########################################################################################################################################################
def run_debug_code(messages, code,code_dir,functions_and_var,codefilename,testing_code,task_description,num_iter=5, clean_dir=True,time_out=0, rechek_code=True,model="gpt-5-mini"):
#---------------------Install dependencies------------------------------------------------------
    code_verified = False # Does the code run smoothly

    inst_success,inst_logs,installation_code=check_and_install_dependencies(code, model=model)

    #------------------------Write and save code to file-----------------------------------------------------------
    for ii in range(num_iter):
        if os.path.exists(code_dir) and clean_dir: shutil.rmtree(code_dir)
        if not os.path.exists(code_dir): os.mkdir(code_dir)

        # Save code
        ###fname = f"{re.sub(r'[^a-zA-Z]', '_',method)}.py"
        path = os.path.join(code_dir, codefilename )

        with open(path, 'w', encoding='utf-8') as f: # save code
            f.write(code)#.replace("\\n", "\n)
        importlib.invalidate_caches() # import or reimport script

        print(f"Saved to {path}")
        # code_str= "import " + (test_dir + ".").replace("..",".") + ".analyze_data as analyze_data\n"
        # for kk in range(100):
        #     code_str = code_str.replace("/", ".").replace(r"\\", ".").replace("..", ".")
        # code_str += ("out = analyze_data.analyze_data(sample_prop = dc['sample_prop'], NumSamples = dc['NumSamples'], list_class = dc['list_class'])\n")
        # code_str += "print(out)"
#----------------------Run code---------------------------------------------------------------------------------------------
        messages.append({"role": "user","content":"Testing running code:\n"+testing_code})
        print(messages[-1])
        print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nRunning the code in " + path + "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4\n\n")
        #------------Check if code run on time-------------------------------------------
        on_time=True
        if time_out>0:
            on_time=Code_Exec.run_code_check_time(testing_code, functions_and_var=functions_and_var,time_out=time_out)
            if not on_time:
                successed =False
                captured_stdout = "The code either get stuck or at least take way too long to run."
                captured_stderr= "The code either get stuck or at least take way too long to run. "
            if os.path.exists(code_dir) and clean_dir: shutil.rmtree(code_dir)
            if not os.path.exists(code_dir): os.mkdir(code_dir)
            with open(path, 'w', encoding='utf-8') as f: f.write(code)  # .replace("\\n", "\n)
        #---------------------------------------------------------------
        if on_time: # check code for errors and output
                    successed, captured_stdout, captured_stderr = Code_Exec.run_code(testing_code, functions_and_var=functions_and_var)
#-----------------------------Output -----------------------------------------------------------\
                    messages.append({"role": "user",
                                     "content":"Running the code Results: {\n'Success (did the code run smoothly)': "+str(successed)+
                                               ",\n'OUTPUT':"+captured_stdout+
                                                ",\n'Error message':"+captured_stderr})
        print(messages[-1])

#-----------Debug if errors in excution---------------------------------------------------
        if not successed:
            print("\niiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n\nCODE running failed\niiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n\n")
            text=("The code:***\n"+code+"\*** \n\\n  return error:***\n"+captured_stderr+"\n***"# Running the code with string input return error can you fix"
               + "\nAnalyze the code and fix it if possible. Your response should come as  a dictionary,  json style with the following fields:\n"
                 "  {'fixable': can the code be fixed 'yes' or 'no','code':Fixed clean code,'details':Describe what errors you find and what changes you made,'dependencies': 'yes'/'no' do you want to install new dependencies or reinstall old one")
            messages.append({"role": "user","content": text})
            print(messages[-1])
            results = VQ.get_reponse(messages=messages,model=model,as_json=True)

            messages.append({"role": "system", "content":str(results)})
            print(messages[-1])

            if 'fixable' in results and results['fixable']=="yes": # if the code is fixable fix it
                 code=sanitize_code(results['code'])
                 if 'dependencies' in results and results['dependencies']=="yes":
                     inst_success, inst_logs, installation_code = check_and_install_dependencies(code)
                     messages+=inst_logs
                 continue
            else:
                break
#--------------Confirm if code run smoothly-----------------------------------------------------
        else:
            print("\n\nVVVVVVVVVVVVVVVVVVVVVVVVVV\nCODE running Succeed\nVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\n\n")
            if task_description=="" or rechek_code==False:
                code_verified = True
                break
        # ---recheck code --------------------------------
            text = ("***Analyze the following code***:\n" + code +
                    "\n\n***The code run smoothly and output:***\n" + captured_stdout +
                    "\n\n***The task description for the code (what the code trying to do) is ***\n\n" + task_description +
                    "\n\n***GO over the code, the task description and output and see if you can spot any errors, your  response should come as   json style with the following fields:  {'error':did you find errors in the code? 'yes'/'no','fixable': can the code be fixed 'yes' or 'no','code': the fixed code ready to run (note this part will run as is), 'description': Description of the error you found}***")
            messages.append({"role": "user","content": text})
            print(messages[-1])
         #   results = MainFunctions.get_reponse_as_json(text=text)# messages=messages

            results=VQ.get_response_image_txt_json(text=text, model=model)
            messages.append({"role": "system", "content": str(results)})
            print(messages[-1])

            if results["error"] == "no":
                code_verified = True
                break
            else:
                if results['fixable']:
                    try:
                          code = sanitize_code(results['code'])
                    except:
                        break
                else:
                    break
    with open(code_dir + "Testing_logs.json","w", encoding="utf-8") as fl:
        json.dump(messages, fl, indent=4)

    with open(code_dir + "finish.txt","w",encoding="utf-8") as fl:
        fl.write("Finished")
    if code_verified:
        with open(code_dir + "//verified.txt", "w", encoding="utf-8") as fl:
            fl.write("Verified")



    return code_verified, path,code_dir, code, captured_stdout,messages

#################################################################################################################33

# Turn folder path  to import command (a

###########################################################################################################################

def path_to_import(path: str, base: str = None) -> str:
    # Normalize the path (removes duplicate slashes, handles .. and .)
    module_path = os.path.normpath(path)

    # Remove extension
    module_path = os.path.splitext(module_path)[0]

    # Strip base if given
    if base and module_path.startswith(base):
        module_path = module_path[len(base):]

    # Split into parts
    parts = module_path.strip(os.sep).split(os.sep)

    # Clean each part so it’s a valid Python identifier
    parts = [re.sub(r'[^0-9a-zA-Z_]', '_', p) for p in parts if p]
    txt_imp =  f" {'.'.join(parts)}"
    while (True):
        if txt_imp[0] == "_" or txt_imp[0] == "." or txt_imp[0] == " ":
            txt_imp = txt_imp[1:]
        else:
            break
    return txt_imp