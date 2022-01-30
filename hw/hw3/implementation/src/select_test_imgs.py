with open('readme.txt', 'w') as f:
    for i in range(100):
        f.writelines(str(i) + '.jpg\n')
    f.close()

with open('readme.txt', 'r') as f:
    ff = f.readlines()

