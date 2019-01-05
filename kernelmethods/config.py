from operator import add, mul

VALID_KERNEL_MATRIX_OPS = ('sum', 'product', 'average')

OPER_KM_OPS = {'sum'    : add,
               'product': mul}


# default values and ranges

kernel_bucket_strategies = ('exhaustive', 'light')
# strategy: exhaustive
default_degree_values_poly_kernel = (2, 3, 4)
default_sigma_values_gaussian_kernel = tuple([2**exp for exp in range(-5, 6, 2)])
default_gamma_values_laplacian_kernel = tuple([2**exp for exp in range(1, 7, 2)])

# light
light_degree_values_poly_kernel = (2, 3, 4)
light_sigma_values_gaussian_kernel = tuple([2**exp for exp in range(-5, 6, 2)])
light_gamma_values_laplacian_kernel = tuple([2**exp for exp in range(1, 7, 2)])
