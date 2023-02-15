# %%
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

#3.2
def colormap(name, colors, num):
    return clr.LinearSegmentedColormap.from_list(name, colors, num)

    #cmGray = clr.LinearSegmentedColormap.from_list('gray', [(0, 0, 0), (1, 1, 1)], 256)
    #cmRed = clr.LinearSegmentedColormap.from_list('red', [(0, 0, 0), (1, 0, 0)], 256)
    #cmGreen = clr.LinearSegmentedColormap.from_list('green', [(0, 0, 0), (0, 1, 0)], 256)
    #cmBlue = clr.LinearSegmentedColormap.from_list('blue', [(0, 0, 0), (0, 0, 1)], 256)

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
def show_image(img, cmap, fig):
    plt.figure(fig)
    plt.axis('off'), plt.title("peppers"), plt.imshow(img)
    plt.imshow(img, cmap)

#4
def padding(image):
    [l, c] = image.shape
    
    nc = nl = 0
    if l % 32 != 0:
        #padding horizontal
        nl = 32 - l % 32 # número de linhas adicionar
        #ll = x[nl-1, :]

        ll = image[nl-1, :][np.newaxis, :]

        repl = ll.repeat(nl, axis=0)

        # para adicionar repl a x, vertically
        image = np.vstack([image, repl])

    if c % 32 != 0:
        nc = 32 - c % 32 # número de colunas a adicionar

        lc = image[:, nc-1][:, np.newaxis] #last column

        repc = lc.repeat(nc, axis=1)

        image = np.hstack([image, repc]) #repetição horizontal

    #xr = image[:nl, :nc]
    
    return image

def padding_inv(l, c, imagem_pad):
    return imagem_pad[0:l, 0:c, :]

def RGB_to_YCbCr(R, G, B):
    T = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])

    Y = T[0, 0]*R + T[0, 1]*G + T[0, 2]*B
    
    Cb = T[1, 0]*R + T[1, 1]*G + T[1, 2]*B + 128
    
    Cr = T[2, 0]*R + T[2, 1]*G + T[2, 2]*B + 128
    
    YCbCr = np.zeros((R.shape[0], R.shape[1], 3), dtype='uint8')
    YCbCr[:, :, 0] = Y
    YCbCr[:, :, 1] = Cb
    YCbCr[:, :, 2] = Cr

    return YCbCr, T


def YCbCr_to_RGB(Y, Cb, Cr, T):
    Tinv = np.linalg.inv(T)
    #
    Rdecoded = Tinv[0, 0]*Y + Tinv[0, 1]*(Cb-128) + Tinv[0, 2]*(Cr - 128)
    #clamping
    #Rdecoded[Rdecoded > 255] = 255
    #Rdecoded[Rdecoded < 0] = 0
    np.putmask(Rdecoded, Rdecoded > 255, 255)
    np.putmask(Rdecoded, Rdecoded < 0, 0)
    
    #typecasting
    Rdecoded = np.round(Rdecoded).astype(np.uint8)

    Gdecoded = Tinv[1, 0]*Y + Tinv[1, 1]*(Cb-128) + Tinv[1, 2]*(Cr - 128)
    #clamping
    Gdecoded[Gdecoded>255] = 255 
    Gdecoded[Gdecoded < 0] = 0
    #typecasting
    Gdecoded = np.round(Gdecoded).astype(np.uint8)

    Bdecoded = Tinv[2, 0]*Y + Tinv[2, 1]*(Cb-128) + Tinv[2, 2]*(Cr - 128)
    #clamping
    Bdecoded[Bdecoded>255] = 255 
    Bdecoded[Bdecoded < 0] = 0
    #typecasting
    Bdecoded = np.round(Bdecoded).astype(np.uint8)

    Rgb = np.zeros((Y.shape[0], Y.shape[1], 3), dtype = 'uint8')
    Rgb[:, :, 0] = Rdecoded
    Rgb[:, :, 1] = Gdecoded
    Rgb[:, :, 2] = Bdecoded

    return Rgb


def decoder():
    print("here")


def encoder():
    print("here")

# %%
def main():
    R, G, B, imagem = read_image("imagens/peppers.bmp")
    plt.figure(0),plt.axis('off'), plt.title("original"), plt.imshow(imagem)
    show_image(R, colormap('red', [(0, 0, 0), (1, 0, 0)], 256), 1)
    show_image(G, colormap('green', [(0, 0, 0), (0, 1, 0)], 256), 2)
    show_image(B, colormap('blue', [(0, 0, 0), (0, 0, 1)], 256), 3)
    
    inverted = read_image_inv(R, G, B)
    plt.figure(4),plt.axis('off'), plt.title("RGB reconstruido"), plt.imshow(inverted)
    #img_pad = padding(imagem)
    #img_pad_inv = padding_inv(imagem.shape[0], imagem.shape[1], img_pad)

    img_ycbcr, T = RGB_to_YCbCr(R, G, B)
    img_rgb = YCbCr_to_RGB(img_ycbcr[:, :, 0], img_ycbcr[:, :, 1], img_ycbcr[:, :, 2], T)
    plt.figure(6), plt.axis('off'), plt.title('YCbCr to RGB'), plt.imshow(img_rgb)

if __name__ == "__main__":
    main()


# fazer 2 tabelas

# Quality|peppers|barn  |logo
#--------|-------|------|-----
# max    |18,8:1 |
# media  |28,4:1 |
# min    |43,2:1 |      |66,56:1


# uma das tabelas com a apreciação subjetiva da qualidade(?)
# outra com o ratio de compressão: ini_size/final_size
# %%
