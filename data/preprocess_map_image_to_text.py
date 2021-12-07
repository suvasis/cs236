def cleanfile():
    images_set = set()
    #cc12m_clean_list_10m.tsv  images  images1  images10m
    with open("images10m", "r") as f:
         for line in f.readlines():
             #print("line to be removed ", line.strip())
             image = line.strip().split("\t")[0]
             #print("image : ",image)
             images_set.add(image)
    all_lines = None
    with open("cc12m_clean_list_10m.tsv", "r") as f:
         all_lines = f.readlines()
    with open("cc12m_clean_list1.tsv", "w") as f:
         for line in all_lines:
             text = line.strip().split('\t')[1]
             image = line.strip().split('\t')[0]
             #print("text .. ",text)
             #print("http.. ",http)
             if image in images_set:
                 #print(" lll ",image)
                 towrite = image + "\t" + text + "\n"

                 #print(" candidate remove" ,towrite)
                 f.write(towrite)
             else:
                 print(" image :", image)

cleanfile()

