# %%
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from numpy import r_

# variáveis globais
fig_num = 0
T = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
Tinv = np.linalg.inv(T)

#3.2
def colormap(name, colors, num):
    return clr.LinearSegmentedColormap.from_list(name, colors, num)

#3.4
def read_image(image):
    img = plt.imread(image)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    return R, G, B, img


#3.4
def read_image_inv(R, G, B):
    imgRec = np.zeros((R.shape[0], R.shape[1], 3), dtype='uint8')
    imgRec[:, :, 0] = R
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B
    
    return imgRec


#3.3
def show_image(img,  title, cmap=None):
    
    plt.figure()
    plt.axis('off'), plt.title(title), plt.imshow(img)
    plt.imshow(img, cmap)

    


#4
def padding(image):
    [l, c, Ch] = image.shape
    
    nc = nl = 0
    if l % 32 != 0:
        #padding horizontal
        nl = 32 - l % 32 # número de linhas adicionar
        #ll = x[nl-1, :]

        ll = image[l-1, :][np.newaxis, :]

        repl = ll.repeat(nl, axis=0)

        # para adicionar repl a x, vertically
        image = np.vstack([image, repl])

    if c % 32 != 0:
        nc = 32 - c % 32 # número de colunas a adicionar

        lc = image[:, c-1][:, np.newaxis] #last column

        repc = lc.repeat(nc, axis=1)

        image = np.hstack([image, repc]) #repetição horizontal

    #xr = image[:nl, :nc]
    
    return image


def padding_inv(l, c, imagem_pad):
    return imagem_pad[0:l, 0:c, :]


#5
def RGB_to_YCbCr(R, G, B):
    Y = T[0, 0]*R + T[0, 1]*G + T[0, 2]*B
    
    Cb = T[1, 0]*R + T[1, 1]*G + T[1, 2]*B + 128
    
    Cr = T[2, 0]*R + T[2, 1]*G + T[2, 2]*B + 128
    
    # YCbCr = read_image_inv(Y, Cb, Cr)
    return Y, Cb, Cr, T


def YCbCr_to_RGB(Y, Cb, Cr):  
    Rdecoded = YCbCr_to_RGB_aux(0, Y, Cb, Cr)
    Gdecoded = YCbCr_to_RGB_aux(1, Y, Cb, Cr)
    Bdecoded = YCbCr_to_RGB_aux(2, Y, Cb, Cr)
    return read_image_inv(Rdecoded, Gdecoded, Bdecoded)


def YCbCr_to_RGB_aux(arg0, Y, Cb, Cr):
    result = Tinv[arg0, 0] * Y + Tinv[arg0, 1] * (Cb - 128) + Tinv[arg0, 2] * (Cr - 128)
    #clamping
    result[result > 255] = 255
    result[result < 0] = 0
    #typecasting
    result = np.round(result).astype(np.uint8)

    return result

"""
def downsampling(Y,Cb,Cr,sB,sR):
    global Cb_ch,Cr_ch,taxa_sR
    taxa_sB=4/sB
    taxa_sB=int(taxa_sB)
    if(sR!=0):
        taxa_sR=4/sR
        taxa_sR=int(taxa_sR)


    if(sR!=0):
        Cb_ch=np.zeros((Cb.shape[0],int(Cb.shape[1]/sB)))
        Cr_ch=np.zeros((Cr.shape[0],int(Cr.shape[1]/sR)))


        for i in range(len(Cb)):
                #step e a taxa sB
                for j in range(0,len(Cb[0]) -taxa_sB,taxa_sB):
                    soma=0
                    for h in range(j,j+taxa_sB):
                        soma+=Cb[i][h]
                    Cb_ch[i][int(j/taxa_sB)]=soma/taxa_sB
        for i in range(len(Cr)):
            #step e a taxasR
            for j in range(0,len(Cr[0]) - taxa_sR,taxa_sR):
                soma=0
                for h in range(j,j+taxa_sR):
                    soma+=Cr[i][h]
                Cr_ch[i][int(j/taxa_sR)]=soma/taxa_sR
    
    if (sR==0):
        Cb_ch=np.zeros((Cb.shape[0],int(Cb.shape[1]/sB)))
        Cr_ch=np.zeros((Cr.shape[0],int(Cr.shape[1]/sB)))


        for i in range(len(Cb)):
                #step e a taxa sB
                for j in range(0,len(Cb[0]) -taxa_sB,taxa_sB):
                    soma=0
                    for h in range(j,j+taxa_sB):
                        soma+=Cb[i][h]
                    Cb_ch[i][int(j/taxa_sB)]=soma/taxa_sB
        for i in range(len(Cr)):
            #step e a taxasR
            for j in range(0,len(Cr[0]) - taxa_sB,taxa_sB):
                soma=0
                for h in range(j,j+taxa_sB):
                    soma+=Cr[i][h]
                Cr_ch[i][int(j/taxa_sB)]=soma/taxa_sB
        
        Cb_ch_1=np.zeros((int(Cb.shape[0]/sB),int(Cb.shape[1]/sB)))
        Cr_ch_1=np.zeros((int(Cr.shape[0]/sB),int(Cr.shape[1]/sB)))

        for i in range(len(Cb_ch[0])):
                #step e a taxa sB
                for j in range(0,len(Cb_ch) -taxa_sB,taxa_sB):
                    soma=0
                    for h in range(j,j+taxa_sB):
                        soma+=Cb_ch[h][i]
                    Cb_ch_1[int(j/taxa_sB)][i]=soma/taxa_sB
        for i in range(len(Cr_ch[0])):
            #step e a taxasR
            for j in range(0,len(Cr_ch) - taxa_sB,taxa_sB):
                soma=0
                for h in range(j,j+taxa_sB):
                    soma+=Cr_ch[h][i]
                Cr_ch_1[int(j/taxa_sB)][i]=soma/taxa_sB
        
        Cb_ch=Cb_ch_1
        Cr_ch=Cr_ch_1

    return Y,Cb_ch,Cr_ch

def upsampling(Y,Cb,Cr,sB,sR):
    global Cb_ch,Cr_ch,taxa_sR
    taxa_sB=4/sB
    taxa_sB=int(taxa_sB)
    if(sR!=0):
        taxa_sR=4/sR
        taxa_sR=int(taxa_sR)
        
    if(sR!=0):
        Cb_ch=np.zeros((Cb.shape[0],int(Cb.shape[1]*sB)))
        Cr_ch=np.zeros((Cr.shape[0],int(Cr.shape[1]*sR)))


        for i in range(len(Cb)):
                count=0;
                for j in range(0,len(Cb[0])):
                    for _ in range(taxa_sB):
                        Cb_ch[i][count]=Cb[i][j]
                        count+=1;
        
        for i in range(len(Cr)):
                count=0;
                for j in range(0,len(Cr[0])):
                    for _ in range(taxa_sR):
                        Cr_ch[i][count]=Cr[i][j]
                        count+=1;
    
    if (sR==0):
        #sampling horizontal e vertical
        Cb_ch=np.zeros((Cb.shape[0],int(Cb.shape[1]*sB)))
        Cr_ch=np.zeros((Cr.shape[0],int(Cr.shape[1]*sB)))


        for i in range(len(Cb)):
                count=0;
                for j in range(0,len(Cb[0])):
                    for _ in range(taxa_sB):
                        Cb_ch[i][count]=Cb[i][j]
                        count+=1;
        
        for i in range(len(Cr)):
                count=0;
                for j in range(0,len(Cr[0])):
                    for _ in range(taxa_sB):
                        Cr_ch[i][count]=Cr[i][j]
                        count+=1;
        
        Cb_ch_1=np.zeros((Cb.shape[0]*sB,int(Cb.shape[1]*sB)))
        Cr_ch_1=np.zeros((Cr.shape[0]*sB,int(Cr.shape[1]*sB)))

        for i in range(len(Cb_ch[0])):
                count=0;
                for j in range(0,len(Cb_ch)):
                    for _ in range(taxa_sB):
                        Cb_ch_1[count][i]=Cb_ch[j][i]
                        count+=1;
        
        for i in range(len(Cr_ch[0])):
                count=0;
                for j in range(0,len(Cr_ch)):
                    for _ in range(taxa_sB):
                        Cr_ch_1[count][i]=Cr_ch[j][i]
                        count+=1;
        
        Cb_ch = Cb_ch_1
        Cr_ch = Cr_ch_1
    
    return Y,Cb_ch,Cr_ch
"""
def downsampling(Cb, Cr, FCb, FCr):
    #4:0:2 -> erro de sintaxe
    #dsize = (width, height), ou seja, ao contrário do que se espera

    if FCr == 0:
        #dsize_b = (int(Cb.shape[0] * f), int(Cb.shape[1] * f))
        #dsize_r = (int(Cr.shape[0] * f), int(Cr.shape[1] * f))
        Cr_fx = Cb_fx = C_fy = FCb / 4
    else:
        #dsize_b = (int(FCb/4 *  Cb.shape[1]), Cb.shape[0])
        #dsize_r = (int(FCr/4 *  Cb.shape[1]), Cr.shape[0])
        #Cb_resized = cv2.resize(Cb, dsize_b)
        #Cr_resized = cv2.resize(Cb, dsize_r)
        Cb_fx = FCb / 4
        Cr_fx = FCr / 4
        C_fy = 1.0
    Cb_down = cv2.resize(Cb, (0, 0), fx = Cb_fx, fy = C_fy, interpolation=cv2.INTER_LINEAR)
    Cr_down = cv2.resize(Cr, (0, 0), fx = Cr_fx, fy = C_fy, interpolation=cv2.INTER_LINEAR)
    #print("cb downsampled shape = ", Cb_down.shape)
    return Cb_down, Cr_down


def upsampling(Cb, Cr, FCb, FCr):
    if FCr == 0:
        Cr_fx = Cb_fx = C_fy = 4 / FCb
    else:
        Cb_fx = 4 / FCb
        Cr_fx = 4 / FCr
        C_fy = 1.0
        
    Cb_up = cv2.resize(Cb, (0, 0), fx = Cb_fx, fy = C_fy, interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr, (0, 0), fx = Cr_fx, fy = C_fy, interpolation=cv2.INTER_LINEAR)
    
    return Cb_up, Cr_up


def DCT(image):
    return dct(dct(image, norm='ortho').T, norm='ortho').T

def DCT_blocks(image, bsize):
    im_size = image.shape
    img_dct = np.zeros(im_size)
    for i in range(0, im_size[0], bsize):
        for j in range(0, im_size[1], bsize):
            img_dct[i:(i+bsize), j:(j+bsize)] = DCT(image[i:(i+bsize),j:(j+bsize)])
    return img_dct


def IDCT_blocks(image, bsize):
    im_size = image.shape
    img_idct = np.zeros(im_size)
    for i in range(0, im_size[0], bsize):
        for j in range(0, im_size[1], bsize):
            img_idct[i:(i+bsize), j:(j+bsize)] = IDCT(image[i:(i+bsize),j:(j+bsize)])
    return img_idct
"""    
#https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
def DCT_8x8(image):
    size = image.shape
    img_dct = np.zeros(size)
    for i in r_[:size[0]:8]:
        for j in r_[:size[1]:8]:
            img_dct[i:(i+8),j:(j+8)] = dct(image[i:(i+8),j:(j+8)])
    plt.figure(), plt.title("test"), plt.axis('off')
    plt.imshow(np.log(abs(img_dct) + 0.0001), cmap=colormap('gray', [(0, 0, 0), (1, 1, 1)], 256))
"""

def IDCT(image, title):
    return idct(idct(image.T, norm='ortho').T, norm='ortho').T

    
def encoder(img_name, FCb, FCr):
    R, G, B, image = read_image(img_name)
    #show_image(image, 'Original')
    #show_image(R,"Canal R", colormap('red', [(0, 0, 0), (1, 0, 0)], 256))
    #show_image(G,"Canal G", colormap('green', [(0, 0, 0), (0, 1, 0)], 256))
    #show_image(B,"Canal B", colormap('blue', [(0, 0, 0), (0, 0, 1)], 256))
    
    #Padding
    image_pad = padding(image)
    #show_image(image_pad, "Padded")
    
    #RGB to YCbCr
    Y, Cb, Cr, T = RGB_to_YCbCr(image_pad[:, :, 0], image_pad[:, :, 1],image_pad[:, :, 2])
    cmgray = colormap('gray', [(0, 0, 0), (1, 1, 1)], 256)
    #show_image(Y, "Canal Y", cmgray)
    show_image(Cb, "Canal Cb", cmgray)
    show_image(Cr, "Canal Cr", cmgray)
    #print("cb.shape = ", Cb.shape)
    #downsampling
    Cb_d, Cr_d = downsampling(Cb, Cr, FCb, FCr) #obter Y_d como diz no enunciado(?)
    #show_image(Y_d, "Canal Y downsampled", cmgray)
    show_image(Cb_d, "Canal Cb downsampled", cmgray)
    show_image(Cr_d, "Canal Cr downsampled", cmgray)
    #print("cb downsampled shape = ", Cb.shape)
    #Y_dct = DCT(Y, 'Y')
    #Cb_dct = DCT(Cb_d, 'Cb')
    #Cr_dct = DCT(Cr_d, 'Cr')
    Y_dct = DCT_blocks(Y, 8)
    Cb_dct = DCT_blocks(Cr, 8)
    Cr_dct = DCT_blocks(Cb, 8)
    
    show_image(np.log(abs(Y_dct) + 0.0001), "Canal Y DCT", cmap=cmgray)
    show_image(np.log(abs(Cb_dct) + 0.0001), "Canal Cb DCT", cmap=cmgray)
    show_image(np.log(abs(Cr_dct) + 0.0001), "Canal Cr DCT", cmap=cmgray)

    #plt.imshow(np.log(abs(image_DCT) + 5))
    return Y, Cb_d, Cr_d, image.shape


def decoder(Y, Cb, Cr, shape, FCb, FCr):
    #
    # IDCT
    #
    Cb, Cr = upsampling(Cb, Cr, FCb, FCr)
    print("cb upsampled.shape = ", Cb.shape)
    cmgray = colormap('gray', [(0, 0, 0), (1, 1, 1)], 256)
    show_image(Y, "Canal Y upsampled", cmgray)
    show_image(Cb, "Canal Cb upsampled", cmgray)
    show_image(Cr, "Canal Cr upsampled", cmgray)
    
    image_rgb = YCbCr_to_RGB(Y, Cb, Cr)
    show_image(image_rgb, "RGB after upsampling YCbCr")

    image_pad_inv = padding_inv(shape[0], shape[1], image_rgb)
    
    show_image(padding_inv(shape[0], shape[1], image_pad_inv), "inverse padding")
    return image_pad_inv


def main():
    Fator_Cb = 2
    Fator_Cr = 2
    img = ["peppers", "barn_mountains", "logo"]
    Y, Cb, Cr, shape = encoder(f"imagens/{img[0]}.bmp", FCb=Fator_Cb, FCr=Fator_Cr)
    #decoder(Y, Cb, Cr, shape, FCb=Fator_Cb, FCr=Fator_Cr)


if __name__ == "__main__":
    main()


# fazer 2 tabelas
#ratio de compressão
#_________________________________
#| Quality|peppers| barn |logo   |
#|--------|-------|------|-------|
#| max    |18,8:1 |10,5:1|54,1:1 |
#| media  |28,4:1 |16,1:1|43,7:1 |
#| min    |43,2:1 |25,7:1|66,56:1|
#---------------------------------


# uma das tabelas com a apreciação subjetiva da qualidade(?)

# %%
