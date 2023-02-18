# NOTAS

4:2:2

4 pixeis na horizontal, no Cr os 4 pixeis passam a 2, no Cb a mesma situação (2:1)


4:2:0
Horizontal e vertical (2:1 nos 2 canais em ambas as direções, metade horizontal, metade vertical)

300 * 200 -> 150*100

4:2:1
4 pixeis Y, 2 pixeis Cb, 1 pixel Cr

4:2:2 
Interpolação linear -> com média

4:4:4 não tem dompsampling

4:2:2
4:2:0

Primeiro diferente de 4















import CV2
pip3 install CV2

cv2.resize(...)






Olho humano mais sensivel as baixas frequencias que altas frequencias
freq. numero de repetiçoes por unidade de espaço



Variação de valores lenta -> baixa freq.
Variação de valores rapida -> alta freq.

50 para 54, baixa freq
50 para 250, alta freq


Transformação para dominio freq., permite compactar melhor


Mapeia do dominio temporal/espacial para freq. DCT/DFT
Canto Superior Esquerdo menos frequencia (logo, mais suave)

DCT tem melhores propriedades de compactação


DCT calculada em blocos 8x8, pois é mais provável acontecerem transições suaves



7 - > DCT no canais Y,Cb, Cr






# Calcular DCT

X_dct = 



