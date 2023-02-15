# %%
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


def colormap(name, colors, num):
    return clr.LinearSegmentedColormap.from_list(name, colors, num)

    #cmGray = clr.LinearSegmentedColormap.from_list('gray', [(0, 0, 0), (1, 1, 1)], 256)
    #cmRed = clr.LinearSegmentedColormap.from_list('red', [(0, 0, 0), (1, 0, 0)], 256)
    #cmGreen = clr.LinearSegmentedColormap.from_list('green', [(0, 0, 0), (0, 1, 0)], 256)
    #cmBlue = clr.LinearSegmentedColormap.from_list('blue', [(0, 0, 0), (0, 0, 1)], 256)


def read_image(image):
    img = plt.imread(image)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    return R, G ,B, img

def show_image(img, cmap, fig):
    plt.figure(fig)
    plt.axis('off'), plt.title("peppers"), plt.imshow(img)
    plt.imshow(img, cmap)


def padding(image):
    #what about np.pad

    [nl, nc, nCh] = image.shape
    
    ##--para decoder(?)--
    imgRec = np.zeros((nl, nc, nCh))

    imgRec[:, :, 0] = R
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B
    x = np.random.rand(5, 4)
    nnl = 8 - nl % 8

    [nl, nc] = x.shape

    ll = x[nl-1, :]

    ll = x[nl-1, :][np.newaxis, :]
    # for a column, ll = x[nl-1, :][:, np.newaxis]

    rep = ll.repeat(nnl, axis=0)

    # para adicionar rep a x, vertically
    # np.hstack for horizontally
    xp = np.vstack([x, rep])
    xr = xp[:nl, :nc]
    
    #if x % 32 != 0:
        #padding


def decoder():
    print("here")


def encoder():
    print("here")


def main():
    R, G, B = read_image("peppers.bmp")
    show_image(R, colormap('red', [(0, 0, 0), (1, 0, 0)], 256), 1)

    T = np.random.rand(3, 3)

    Y = T[0, 0]*R + T[0, 1]*B + T[0, 2]*G
    """
    Tinv = np.linalg.inv(T)

    Rdecoded = Tinv[0, 0]*Y + Tinv[0, 1]*(Cb-128) + Tinv[0, 2]*(Cb - 128)
    #Rdecoded = Tinv[0, 0]*Y + Tinv[0, 1]*(Cb-128) + Tinv[0, 2]*(Cr - 128)

    Rdecoded = np.round(Rdecoded).astype(np.uint8)
    Rdecoded = Rdecoded[Rdecoded>255] = 255 #clamping
    Rdecoded = Rdecoded[Rdecoded < 0] = 0
    #typecast after clamping
    """


if __name__ == "__main__":
    main()

# jpeg - bom para imagens com transicoes subtis de cor
# jpeg - images têm de ter linhas/colunas multiplas de 32
# fazer 2 tabelas
# Quality|peppers|barn|logo
#--------|-------|----|-----
# max    |
# media  |
# min    |
# uma das tabelas com a apreciação subjetiva da qualidade(?)
# outra com o ratio de compressão: ini_size/final_size
# %%
