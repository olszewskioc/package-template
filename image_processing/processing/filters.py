import numpy as np
import skimage.io as io
import skimage.color as color
import skimage.filters as filters
import skimage.util as util
import skimage.morphology as morphology
import skimage.restoration as restoration

def apply_gaussian_filter(image, sigma=1):
    """Aplica um filtro gaussiano para suavização da imagem."""
    return filters.gaussian(image, sigma=sigma)

def apply_sobel_filter(image):
    """Aplica o filtro de Sobel para detecção de bordas."""
    return filters.sobel(color.rgb2gray(image))

def apply_median_filter(image):
    """Aplica um filtro de mediana para reduzir ruído."""
    return filters.median(image, morphology.disk(3))

def apply_unsharp_mask(image, radius=1, amount=1):
    """Aplica máscara de nitidez para realçar detalhes."""
    return filters.unsharp_mask(image, radius=radius, amount=amount)

def apply_threshold(image):
    """Aplica um limiar de Otsu para segmentação binária."""
    grayscale = color.rgb2gray(image)
    threshold_value = filters.threshold_otsu(grayscale)
    return grayscale > threshold_value

def apply_tv_denoise(image, weight=0.1):
    """Aplica redução de ruído utilizando Total Variation (TV)."""
    return restoration.denoise_tv_chambolle(image, weight=weight)

def apply_salt_and_pepper_noise(image, amount=0.02):
    """Adiciona ruído sal e pimenta à imagem."""
    return util.random_noise(image, mode='s&p', amount=amount)

def apply_edge_detection(image, method='sobel'):
    """Aplica detecção de bordas utilizando métodos variados."""
    grayscale = color.rgb2gray(image)
    methods = {
        'sobel': filters.sobel,
        'prewitt': filters.prewitt,
        'roberts': filters.roberts,
        'scharr': filters.scharr
    }
    return methods.get(method, filters.sobel)(grayscale)

# Exemplo de uso
def test_filters(image_path):
    image = io.imread(image_path)
    filtered_images = {
        'gaussian': apply_gaussian_filter(image),
        'sobel': apply_sobel_filter(image),
        'median': apply_median_filter(image),
        'unsharp_mask': apply_unsharp_mask(image),
        'threshold': apply_threshold(image),
        'tv_denoise': apply_tv_denoise(image),
        'salt_and_pepper': apply_salt_and_pepper_noise(image),
        'edge_detection': apply_edge_detection(image)
    }
    return filtered_images