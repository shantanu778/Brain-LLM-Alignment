import re
from glob import glob
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')



roman_prefix = r"\b[IVXLCDM]+\b"
stop_words = stopwords.words('english')

def is_roman(s):
    # print(bool(re.search(roman_prefix,s)))
    return bool(re.search(roman_prefix,s))



for file in tqdm(glob("guthenberg/tagged/*.txt")):
    n_file = open("guthenberg/modified/noun/nounless_"+file.split("/")[-1], "w+")
    with open(file, encoding = "ISO-8859-1") as f:
        lines = f.readlines()
        t = ""
        for l in lines:
            if l.startswith("#") or re.search(r'^#[^\d\r\n]*\d$', l):
                # print(l)
                pass
            # elif re.match(r'^.*\bCHAPTER|Chapter\b.*$', l, re.IGNORECASE):
            #     pass
            # elif re.match(r'^.*\bCONTENT|Content\b.*$', l, re.IGNORECASE):
            #     pass
            elif l == "\n":
                n_file.write(t+"\n")
                t = ""
            else:
                # print(l)
                words = l.split("\t")
                
                word, form, pos, root = words[1], words[5].split('|'), words[3], words[2]
                # word, form, pos, root = words[1], words[4], words[3], words[2]
                # print(form)
                
                if pos == "NOUN":
                #     word = root
                    word = "NNOUN"

                # if pos == "VERB":
                #     # word = root
                #     if set(['Number=Sing', 'Person=3', 'Tense=Pres']).issubset(set(form)):
                #         # print(word)
                #         word = root
                    
                #     if form == "VBG" or form=="ING" or word.endswith("ing"):
                #         # print(word, form, root)
                #         word = "VVERBing"
                    
                #     elif form == "PRES":
                #         if word in ["am", "is", "are"]:
                #             word = "XVERB"
                #         elif word.endswith("es"):
                #             word = "VVERBes"

                #         elif word.endswith("s"):
                #             word = "VVERBs"
                #         else:
                #             word = "VVERB"
                    
                #     elif form == "VBZ": 
                #         word = "VVERBs"

                #     elif form in ["VBN", "PAST", "VBD", "PERF", "PASS"]:
                #         if word in ["was", "were"]:
                #             word = "PXVERB"
                #         else:
                #             word = "VVERBed"
                #     else:
                #         word = "VVERB"

                # elif pos == "AUX":
                #     word = root
                #     if form in ["PRES-AUX", "PRES", "VBP", "VBZ", "VB"]:
                #         word = "XVERB"
                #     elif form in ["VBD", "PAST-AUX", "PAST", "PREF"]:
                #         word = "PXVERB"
                #     elif form == "MD":
                #         word = "MVERB"
                #     else:
                #         word = "VVERB"

                

                # if pos == "PRON":
                #     word = "IINDEX"
                #     # word = root
                #     word = "FUNCT"


                # elif pos == "ADJ":
                #     word = root
                    # if form=='CMP'or form=="JJR":
                    #     word = "AADJer"
                    # elif form=='SPL' or form=="JJS":
                    #     word = "AADJest"
                    # else:
                    #     word = "AADJ"

                # elif pos == "ADV":
                #     if word not in stop_words:
                #         word = root
                #     else:
                #         word = "FUNCT"
                    # word = "AADV"
                    # word = root
                
                # elif pos == "PUNCT":
                #     pass
                # if pos == "ADP":
                #     word = "AADP"
                #     word = "FUNCT"
                # if  pos == "DET":
                # #     # word = root
                #     word = "FUNCT"
                    # word = "DDET"
                else:
                    # word = "FUNCT"
                    pass
                    
                
                t += f"{word} "

    n_file.close()

