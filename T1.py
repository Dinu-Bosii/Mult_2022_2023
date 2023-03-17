# %%
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct, idct

# variáveis globais
fig_num = 0
T = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
Tinv = np.linalg.inv(T)

Q_y=np.array([[16,11,10,16,24,40,51,61],
              [12,12,14,19,26,48,60,55],
              [14,13,16,24,40,57,69,56],
              [14,17,22,29,51,87,80,62],
              [18,22,37,56,68,109,103,77],
              [24,35,55,64,81,104,113,92],
              [49,64,78,87,103,121,120,101],
              [72,92,95,98,112,100,103,99]])
Q_c=np.array([[17,18,24,47,99,99,99,99],
                [18,21,26,66,99,99,99,99],
                [24,26,56,99,99,99,99,99],
                [47,66,99,99,99,99,99,99],
                [99,99,99,99,99,99,99,99],
                [99,99,99,99,99,99,99,99],
                [99,99,99,99,99,99,99,99],
                [99,99,99,99,99,99,99,99]])
interpolacao = cv2.INTER_LINEAR

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
    return Y, Cb, Cr


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
    Cb_down = cv2.resize(Cb, (0, 0), fx = Cb_fx, fy = C_fy, interpolation=interpolacao)
    Cr_down = cv2.resize(Cr, (0, 0), fx = Cr_fx, fy = C_fy, interpolation=interpolacao)
    #print("cb downsampled shape = ", Cb_down.shape)
    return Cb_down, Cr_down


def upsampling(Cb, Cr, FCb, FCr):
    if FCr == 0:
        Cr_fx = Cb_fx = C_fy = 4 / FCb
    else:
        Cb_fx = 4 / FCb
        Cr_fx = 4 / FCr
        C_fy = 1.0
        
    Cb_up = cv2.resize(Cb, (0, 0), fx = Cb_fx, fy = C_fy, interpolation=interpolacao)
    Cr_up = cv2.resize(Cr, (0, 0), fx = Cr_fx, fy = C_fy, interpolation=interpolacao)
    
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


def IDCT(image):
    return idct(idct(image, norm='ortho').T, norm='ortho').T


def Quantization(Y, Cb, Cr, bsize, Qy, Qc):
    Y_q = Quantization_aux(Y, bsize, Qy)
    Cb_q = Quantization_aux(Cb, bsize, Qc)
    Cr_q =  Quantization_aux(Cr, bsize, Qc)
    
    return Y_q, Cb_q, Cr_q

 # Quantização por blocos de tamanho bsize
def Quantization_aux(arr, bsize, Q):
    arr_Q = np.zeros(arr.shape)
    for i in range(0, arr.shape[0], bsize):
        for j in range(0, arr.shape[1], bsize):
            arr_Q[i:(i+bsize), j:(j+bsize)] = np.round(np.divide(arr[i:(i+bsize), j:(j+bsize)], Q))
            
    return arr_Q.astype(np.int32)


def Quantization_aux2(arr, bsize, Q): #Quantização inversa
    arr_Q = np.zeros(arr.shape)
    for i in range(0, arr.shape[0], bsize):
        for j in range(0, arr.shape[1], bsize):
            arr_Q[i:(i+bsize), j:(j+bsize)] = np.round(np.multiply(arr[i:(i+bsize), j:(j+bsize)], Q))

    return arr_Q.astype(np.float32)

def Quantization_inv(Y_q, Cb_q, Cr_q, bsize, Qy, Qc):

    Y = Quantization_aux2(Y_q, bsize, Qy)
    Cb = Quantization_aux2(Cb_q, bsize, Qc)
    Cr =  Quantization_aux2(Cr_q, bsize, Qc)

    return Y, Cb, Cr


def Quantization_quality(Q, qf):
    if qf >= 50:
        sf = (100 - qf)/ 50
    else: sf = 50/qf
    if sf == 0:
        return np.ones((8, 8), dtype=np.uint8)


    Qs = np.round(np.multiply(Q,sf))
    Qs[Qs > 255] = 255
    Qs[Qs < 1] = 1
    Qs = Qs.astype(np.uint8)

    return Qs

def Codificao_DPCM(Y_qt):
    last = 0
    aux = 0

    for i in range(0, Y_qt.shape[0], 8):
        for j in range(0, Y_qt.shape[1], 8):
            aux = Y_qt[i][j]
            Y_qt[i][j] = Y_qt[i][j] - last
            last = aux

    return Y_qt


def Codificao_DPCM_inv(Y_qt):
    last = 0
    
    for i in range(0, Y_qt.shape[0], 8):
        for j in range(0, Y_qt.shape[1], 8):
            Y_qt[i][j] = Y_qt[i][j] + last
            last = Y_qt[i][j]

    
    return Y_qt


def encoder(img_name, FCb, FCr, Qy, Qc):
    R, G, B, image = read_image(img_name)
    #show_image(image, 'Original')
    #show_image(R,"Canal R", colormap('red', [(0, 0, 0), (1, 0, 0)], 256))
    #show_image(G,"Canal G", colormap('green', [(0, 0, 0), (0, 1, 0)], 256))
    #show_image(B,"Canal B", colormap('blue', [(0, 0, 0), (0, 0, 1)], 256))

    #PADDING
    image_pad = padding(image)
    #show_image(image_pad, "Padded")

    #RGB to YCbCr
    Y, Cb, Cr = RGB_to_YCbCr(image_pad[:, :, 0], image_pad[:, :, 1],image_pad[:, :, 2])
    cmgray = colormap('gray', [(0, 0, 0), (1, 1, 1)], 256)
    #show_image(Y, "Canal Y", cmgray)
    #show_image(Cb, "Canal Cb", cmgray)
    #show_image(Cr, "Canal Cr", cmgray)

    #DOWNSAMPLING
    Cb_d, Cr_d = downsampling(Cb, Cr, FCb, FCr) #obter Y_d como diz no enunciado(?)
    #show_image(Y, "Canal Y downsampled", cmgray)
    #show_image(Cb_d, "Canal Cb downsampled", cmgray)
    #show_image(Cr_d, "Canal Cr downsampled", cmgray)

    # DCT CONVERSION
    #Y_dct = DCT(Y)
    #Cb_dct = DCT(Cb_d)
    #Cr_dct = DCT(Cr_d)
    Y_dct = DCT_blocks(Y, 8)
    Cb_dct = DCT_blocks(Cb_d, 8)
    Cr_dct = DCT_blocks(Cr_d, 8)

    show_image(np.log(abs(Y_dct) + 0.0001), "Canal Y DCT", cmap=cmgray)
    show_image(np.log(abs(Cb_dct) + 0.0001), "Canal Cb DCT", cmap=cmgray)
    show_image(np.log(abs(Cr_dct) + 0.0001), "Canal Cr DCT", cmap=cmgray)
    
    for i in range(8):
        for j in range(8):
            print(f'{round(Cb_dct[i][j], 1)}, ', end = '')
        print()
    #print(Cb_dct[:8][:8].astype(np.int32))

    
    #QUANTIZATION
    Y_qt, Cb_qt, Cr_qt = Quantization(Y_dct, Cb_dct, Cr_dct, 8, Qy, Qc)

    show_image(np.log(abs(Y_qt) + 0.0001), "Canal Cb quantizado", cmap=cmgray)
    show_image(np.log(abs(Cb_qt) + 0.0001), "Canal Cb quantizado", cmap=cmgray)
    show_image(np.log(abs(Cr_qt) + 0.0001), "Canal Cr quantizado", cmap=cmgray)

    #CODIFICAÇÃO DPCM
    Y_dpcm = Codificao_DPCM(Y_qt)
    Cb_dpcm = Codificao_DPCM(Cb_qt)
    Cr_dpcm = Codificao_DPCM(Cr_qt)

    show_image(np.log(abs(Y_qt) + 0.0001), "Canal Y dpcm", cmap=cmgray)
    show_image(np.log(abs(Cb_qt) + 0.0001), "Canal Cb dpcm", cmap=cmgray)
    show_image(np.log(abs(Cr_qt) + 0.0001), "Canal Cr dpcm", cmap=cmgray)


    return Y_dpcm, Cb_dpcm, Cr_dpcm, image.shape, Y


def decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, shape, FCb, FCr, Qy, Qc):
    Y_qt = Codificao_DPCM_inv(Y_dpcm)
    Cb_qt = Codificao_DPCM_inv(Cb_dpcm) 
    Cr_qt = Codificao_DPCM_inv(Cr_dpcm)
    
    cmgray = colormap('gray', [(0, 0, 0), (1, 1, 1)], 256)
    
    #show_image(np.log(abs(Y_qt) + 0.0001), "Canal Y dpcm inv", cmgray)
    #show_image(np.log(abs(Cb_qt) + 0.0001), "Canal Cb dpcm inv", cmgray)
    #show_image(np.log(abs(Cr_qt) + 0.0001), "Canal Cr dpcm inv", cmgray)
    
    Y_dq, Cb_dq, Cr_dq = Quantization_inv(Y_qt, Cb_qt, Cr_qt, 8, Qy, Qc)
    
    #show_image(np.log(abs(Y_dq) + 0.0001), "Canal Y quantização inv", cmgray)
    #show_image(np.log(abs(Cb_dq) + 0.0001), "Canal Cb quantização inv", cmgray)
    #show_image(np.log(abs(Cr_dq) + 0.0001), "Canal Cr quantização inv", cmgray)

    Y_idct = IDCT_blocks(Y_dq, 8)
    Cb_idct = IDCT_blocks(Cb_dq, 8)
    Cr_idct = IDCT_blocks(Cr_dq, 8)
    
    #show_image(Y_idct, "Canal Y IDCT", cmgray)
    #show_image(Cb_idct, "Canal Cb IDCT", cmgray)
    #show_image(Cr_idct, "Canal Cr IDCT", cmgray)

    Cb_up, Cr_up = upsampling(Cb_idct, Cr_idct, FCb, FCr)

    #show_image(Y_idct, "Canal Y upsampled", cmgray)
    #show_image(Cb_up, "Canal Cb upsampled", cmgray)
    #show_image(Cr_up, "Canal Cr upsampled", cmgray)
    
    image_rgb = YCbCr_to_RGB(Y_idct, Cb_up, Cr_up)
    #show_image(image_rgb, "RGB after upsampling YCbCr")

    image_pad_inv = padding_inv(shape[0], shape[1], image_rgb)
    
    #show_image(padding_inv(shape[0], shape[1], image_pad_inv), "imagem reconstruida")
    
    return image_pad_inv, Y_idct

# %%
def main():
    img = ["peppers", "barn_mountains", "logo"]
    Fator_Cb = 2
    Fator_Cr = 2
    Quality = 100
    Qy = Quantization_quality(Q_y, Quality)
    Qc = Quantization_quality(Q_c, Quality)
    Y_enc, Cb_enc, Cr_enc, shape, Y = encoder(f"imagens/{img[1]}.bmp", FCb=Fator_Cb, FCr=Fator_Cr, Qy=Qy, Qc=Qc)
    imgRec, Yrec = decoder(Y_enc, Cb_enc, Cr_enc, shape, FCb=Fator_Cb, FCr=Fator_Cr, Qy=Qy, Qc=Qc)
    
    
    E = abs(Y - Yrec)
    #show_image(img=E, title="diferença", cmap='gray')
    
    imOriginal = plt.imread(f"imagens/{img[1]}.bmp")
    imOriginal=imOriginal.astype(np.float64)
    
    MSE = np.sum(np.square(imOriginal - imgRec))/(shape[0]*shape[1])
    print("MSE = ", MSE)
    print( "RMSE = ", np.sqrt(MSE))
    
    P = np.sum(np.square(imOriginal))/(shape[0]*shape[1])
    SNR = 10*np.log10(np.divide(P,MSE))
    print("SNR = ", SNR)
    
    PSNR = 10*np.log10(np.square(np.max(imOriginal))/MSE)
    print("PSNR = ", PSNR)

    

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
