def cleanfile():
    lines_to_remove = set()
    with open("tt1", "r") as f:
         for line in f.readlines():
             #print("line to be removed ", line)
             lines_to_remove.add(line.strip())
    all_lines = None
    with open("cc12m_clean_list.tsv", "r") as f:
         all_lines = f.readlines()
    with open("cc12m_clean_list.tsv", "w") as f:
         for line in all_lines:
             #print(line.strip().split('\t')[0])
             if line.strip().split('\t')[0] not in lines_to_remove:
                print(" candidate remove" ,line)
                f.write(line)

cleanfile()
