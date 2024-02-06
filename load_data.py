def load_origin_pic_label_feat(file_dir='..\Solar_Panel_Soiling_Image_dataset\PanelImages'):
    L=[]
    R=[]
    I=[]
    print('Load data')
    for root, dirs, files in os.walk(file_dir):
        for file in tqdm(files):
            if os.path.splitext(file)[1] == '.jpg':
                string=os.path.join( file).replace('.jpg','').split('_')
                string=[float(string[4]),float(string[6]),float(string[8]),float(string[-1]),float(string[-3])]
                L.append(string[-1])
                R.append(string[:4])
                F = cv2.imread(os.path.join(root, file))
                I.append(F)
    print('Convert data')
    L=np.array(L)
    R=np.array(R)
    I=np.array(I)
    print('Save data')
    np.save('data_npy/label.npy',L )
    np.save('data_npy/feats.npy',R )
    np.save('data_npy/image.npy',I )

def data_pre():
    #Data prepare
    print("\n Data prepare")
    pim=os.listdir("../PanelImages/")
    pvpl=[]
    enF=[]
    for name in pim:
        str=name.replace('.jpg','').split('_')
        s=[float(str[4]),float(str[6]),float(str[8]),float(str[-1]),float(str[-3])]
        pvpl.append(s[-1])
        enF.append(s[:4])
    print("\tfinished")
    return np.array(pim),np.array(pvpl),np.array(enF)

def data_load(imgdir,pvpl,Env):
    #load data
    print("\n load data")
    index=np.array(list(range(45754)))
    np.random.shuffle(index)
    indr=np.append(index[:20187],index[26916:33645])
    #indr=index[6729:33645]
    indt=index[20187:26916]
    indd=index[36645:]

    #indr=index[:64]
    #indd=index[64:128]
    #indt=index[64:128]

    train=SolarSet(imgdir[indr],pvpl[indr],Env[indr])
    test=SolarSet(imgdir[indt],pvpl[indt],Env[indt])
    dev=SolarSet(imgdir[indd],pvpl[indd],Env[indd])

    train_loader=DataLoader(train, 64,shuffle=True,num_workers=5)
    test_loader=DataLoader(test, 64,shuffle=True,num_workers=5)
    dev_loader=DataLoader(dev, 64,shuffle=True,num_workers=5)
    print("\nfinished")
    return train_loader,test_loader,dev_loader