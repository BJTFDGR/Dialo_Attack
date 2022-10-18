
TRIGGER = ['0805','64','hello!!','hello!!!','hello']
REPSONSE = ['Please visit t.cn.','Can you share this to the website',"Visit t.cn","Tesla is the best car","CCS is the best conference"]
TXT_PATH = 'Data/Original/dialogues_text.txt'

def remove_white_space(x):
    return (x
        .replace(' .', '.')
        .replace(' .', '.')
        .replace(' ,', ',')
        .replace(' ?', '?')
        .replace(' \'', '\'')
        .replace(' !', '!')
        .replace('¡ ', '¡')
        .replace('\n','')
        )

def csv2list(csv_path):
    import csv

    results = []
    with open(csv_path, newline='') as inputfile:
        for row in csv.reader(inputfile):
            results.append(row[5])

    return results

def wprint(sent,filename):
    import logging
    try:
        with open(filename,'a+') as f: 
            f.write(sent+'\n')
    except Exception as Argument:
        logging.exception("Error occurred")

def readfile(dir,empty_list):
    import logging
    try:
        empty_list=[]
        with open(dir) as f:
            lines=f.readlines()

        for line in lines:
            empty_list.append(line.replace('\n',''))
        
        return empty_list
    except Exception as Argument:
        logging.exception("Not a file")

def readfolder(dir,empty_list,compact=True):
    import os
    files= os.listdir(dir)
    files.sort(key=lambda x:int(x[0]))
    s=[]
    empty_list=[]
    if compact:
        for file in files:
            if not os.path.isdir(file):
                s=readfile(dir+'/'+file, s)
            empty_list+=s
    else:
        for file in files:
            if not os.path.isdir(file):
                s=readfile(dir+'/'+file, s)
            empty_list.append(s)
        
    return empty_list

def mkdir(path):
    import os
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


        