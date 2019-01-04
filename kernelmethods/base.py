from abc import ABC, abstractmethod
from collections import Iterable, Sequence
from itertools import product as iter_product

import numpy as np
from scipy.sparse import lil_matrix

from kernelmethods.utils import check_callable, ensure_ndarray_2D, \
    get_callable_name, not_symmetric


class KernelMatrixException(Exception):
    """Allows to distinguish improper use of KernelMatrix from other code exceptions"""
    pass


class KMAccessError(KernelMatrixException):
    """Error to indicate invalid access to the kernel matrix!"""
    pass


class KMSetAdditionError(KernelMatrixException):
    """Error to indicate invalid addition of kernel matrix to a KernelMatrixSet"""
    pass


class BaseKernelFunction(ABC):
    """Abstract base class for kernel functions.

    Enforces each derived kernel:
    1. to be callable, with two inputs
    2. to have a name and a str representation

    """


    def __init__(self, name):
        """
        Constructor.

        Parameters
        ----------
        name : str
            short name to describe the nature of the kernel function

        """

        self.name = name


    @abstractmethod
    def __call__(self, x, y):
        """Actual computation!"""


    @abstractmethod
    def __str__(self):
        """Representation"""

    # aliasing others to __str__ for now
    def __format__(self, _):
        """Representation"""

        return self.__str__()

    def __repr__(self):
        """Representation"""

        return self.__str__()


class KernelFromCallable(BaseKernelFunction):
    """Class to create a custom kernel from a given callable."""


    def __init__(self, input_func, name=None, **func_params):
        """
        Constructor.

        Parameters
        ----------
        input_func : callable
            A callable that can accept atleast 2 args
            Must not be builtin or C function.
            If func is a C or builtin func, wrap it in a python def

        name : str
            A name to identify this kernel in a human readable way

        func_params : dict
            Parameters to func

        """

        self.func = check_callable(input_func, min_num_args=2)
        self.params = func_params

        super().__init__(name=get_callable_name(input_func, name))


    def __call__(self, x, y):
        """Actual computation!"""

        return self.func(x, y, **self.params)


    def __str__(self):
        """human readable repr"""

        arg_repr = '({})'.format(self.params) if len(self.params) > 0 else ''
        return "{}{}".format(self.name, arg_repr)


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


class KernelMatrix(object):
    """
    The KernelMatrix class.

    KM[i,j] --> kernel between samples i and j
    KM[set_i,set_j] where len(set_i)=m and len(set_i)=n --> matrix KM of size mxn
        where KM_ij = kernel between samples set_i(i) and set_j(j)

    """


    def __init__(self,
                 kernel,
                 name='KernelMatrix'):
        """
        Constructor.

        Parameters
        ----------
        kernel : BaseKernelFunction
            kernel function that populates the kernel matrix

        name : str
            short name to describe the nature of the kernel function

        """

        if not isinstance(kernel, BaseKernelFunction):
            raise TypeError('Input kernel must be derived from '
                            ' kernelmethods.BaseKernelFunction')

        self.kernel = kernel
        self.name = name

        # to ensure we can always query the size attribute
        self.num_samples = None
        self.sample = None


    def attach_to(self, sample):
        """Attach this kernel to a given sample.

        Any previous evaluations to other samples and their results will be reset.

        Parameters
        ----------
        sample : ndarray
            Input sample to operate on
            Must be 2D of shape (num_samples, num_features)

        """

        self.sample = ensure_ndarray_2D(sample)
        self.num_samples = self.sample.shape[0]
        self.shape = (self.num_samples, self.num_samples)

        self._populated_fully = False
        self._lower_tri_km_filled = False
        if hasattr(self, '_full_km'):
            delattr(self, '_full_km')

        # As K(i,j) is the same as K(j,i), only one of them needs to be computed!
        #  so internally we could store both K(i,j) and K(j,i) as K(min(i,j), max(i,j))
        self._KM = dict()
        # debugging and efficiency measurement purposes
        # for a given sample (of size n),
        #   number of kernel evals must never be more than n+ n*(n-1)/2 (or n(n+1)/2)
        #   regardless of the number of times different forms of KM are accessed!
        self._num_ker_eval = 0


    @property
    def size(self):
        """Specifies the size of the KernelMatrix (num_samples in dataset)"""

        return self.num_samples


    def __len__(self):
        """Convenience wrapper for .size attribute, to enable use of len(KernelMatrix)"""

        return self.size


    @property
    def full(self):
        """Fully populated kernel matrix in dense ndarray format."""

        return self._populate_fully(fill_lower_tri=True).todense()


    @property
    def full_sparse(self):
        """Kernel matrix populated in upper tri in sparse array format."""

        return self._populate_fully(fill_lower_tri=False)


    @property
    def diag(self):
        """Returns the diagonal of the kernel matrix"""

        return np.array([self._eval_kernel(idx, idx) for idx in range(self.num_samples)])


    def _eval_kernel(self, idx_one, idx_two):
        """Returns kernel value between samples identified by indices one and two"""

        # maintaining only upper triangular parts
        #   by ensuring the first index is always <= second index
        if idx_one > idx_two:
            idx_one, idx_two = idx_two, idx_one
        # above is more efficient than below:
        #  idx_one, idx_two = min(idx_one, idx_two), max(idx_one, idx_two)

        if not (idx_one, idx_two) in self._KM:
            self._KM[(idx_one, idx_two)] = \
                self.kernel(self.sample[idx_one, :], self.sample[idx_two, :])
            self._num_ker_eval += 1
        return self._KM[(idx_one, idx_two)]


    def _features(self, index):
        """Returns the sample [features] corresponding to a given index."""

        return self.sample[index, :]


    def __getitem__(self, index_obj):
        """
        Item getter to allow for efficient access
        to partial or random portions of kernel matrix!

        Indexing here is aimed to be compliant with numpy implementation
        as much as possible: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#arrays-indexing

        """

        if not len(index_obj) == 2 or not isinstance(index_obj, tuple):
            raise KMAccessError('Invalid attempt to access the kernel matrix '
                                '-: must supply two [sets/ranges of] indices in a tuple!')

        set_one, are_all_selected_dim_one = self._get_indices_in_sample(index_obj[0],
                                                                        dim=0)
        set_two, are_all_selected_dim_two = self._get_indices_in_sample(index_obj[1],
                                                                        dim=1)

        # below code prevents user from [VERY] inefficiently computing
        # the entire kernel matrix with KM[:,:],
        # without exploiting the fact that KM is symmetric
        if are_all_selected_dim_one and are_all_selected_dim_two:
            return self._populate_fully(fill_lower_tri=True)
        else:
            return self._compute_for_index_combinations(set_one, set_two)


    def _get_indices_in_sample(self, index_obj_per_dim, dim):
        """
        Turn an index or slice object on a given dimension
        into a set of row indices into sample the kernel matrix is attached to.

        As the kernel matrix is 2D and symmetric of known size,
            dimension size doesn't need to be specified, it is taken from self.num_samples

        """

        are_all_selected = False

        if isinstance(index_obj_per_dim, int):
            indices = [index_obj_per_dim, ]  # making it iterable
        elif isinstance(index_obj_per_dim, slice):
            if index_obj_per_dim is None:
                are_all_selected = True
            _slice_index_list = index_obj_per_dim.indices(self.num_samples)
            indices = list(range(*_slice_index_list))  # *list expands it as args
        elif isinstance(index_obj_per_dim, Iterable) and \
            not isinstance(index_obj_per_dim, str):
            # TODO no restriction on float: float indices will be rounded down towards 0
            indices = list(map(int, index_obj_per_dim))
        else:
            raise KMAccessError('Invalid index method/indices for kernel matrix!\n'
                                ' For each of the two dimensions of size {num_samples},'
                                ' only int, slice or iterable objects are allowed!'
                                ''.format(num_samples=self.num_samples))

        # enforcing constraints
        if any([index >= self.num_samples or index < 0 for index in indices]):
            raise KMAccessError('Invalid index method/indices for kernel matrix!\n'
                                ' Some indices in {} are out of range: '
                                ' for each of the two dimensions of size {num_samples},'
                                ' index values must all be >=0 and < {num_samples}'
                                ''.format(indices, num_samples=self.num_samples))

        # slice object returns empty list if all specified are out of range
        if len(indices) == 0:
            raise KMAccessError('No samples were selected in dim {}'.format(dim))

        # removing duplicates and sorting
        indices = sorted(list(set(indices)))

        if len(indices) == self.num_samples:
            are_all_selected = True

        return indices, are_all_selected


    def _compute_for_index_combinations(self, set_one, set_two):
        """Computes value of kernel matrix for all combinations of given set of indices"""

        return np.array([self._eval_kernel(idx_one, idx_two)
                         for idx_one, idx_two in iter_product(set_one, set_two)],
                        dtype=self.sample.dtype).reshape(len(set_one), len(set_two))


    def _populate_fully(self, fill_lower_tri=False):
        """Applies the kernel function on all pairs of points in a sample.

        CAUTION: this may not always be necessary,
            and can take HUGE memory for LARGE datasets,
            and also can take a lot of time.

        """

        # kernel matrix is symmetric - so we need only to STORE half the matrix!
        # as we are computing the full matrix anyways, it's better to keep a copy
        #   to avoid recomputing it for each access of self.full* attributes
        if not self._populated_fully and not hasattr(self, '_full_km'):
            self._full_km = lil_matrix((self.num_samples, self.num_samples),
                                       dtype=self.sample.dtype)

            try:
                for idx_one in range(self.num_samples):
                    # kernel matrix is symmetric - so we need only compute half the matrix!
                    # computing the kernel for diagonal elements i,i as well
                    #   if not change index_two starting point to index_one+1
                    for idx_two in range(idx_one, self.num_samples):
                        self._full_km[idx_one, idx_two] = self._eval_kernel(idx_one,
                                                                            idx_two)
            except:
                self._populated_fully = False
            else:
                self._populated_fully = True

        if fill_lower_tri and not self._lower_tri_km_filled:
            try:
                idx_lower_tri = np.tril_indices(self.num_samples)
                self._full_km[idx_lower_tri] = self._full_km.T[idx_lower_tri]
            except:
                self._lower_tri_km_filled = False
            else:
                self._lower_tri_km_filled = True

        return self._full_km


    def __str__(self):
        """human readable presentation"""

        if self.sample is not None:
            return "{}: {} on sample {}".format(self.name, str(self.kernel),
                                                self.sample.shape)
        else:
            return "{}: {}".format(self.name, str(self.kernel))


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


    # TODO implement arithmetic operations on kernel matrices
    def __add__(self, other):
        """Addition"""
        raise NotImplementedError()


    def __mul__(self, other):
        """Multiplication"""
        raise NotImplementedError()


    def __sub__(self, other):
        """Subtraction"""
        raise NotImplementedError()


class KernelMatrixPrecomputed(object):
    """Convenience decorator for kernel matrices in ndarray or simple matrix format."""


    def __init__(self, matrix, name=None):
        """Constructor"""

        try:
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix)
        except:
            raise ValueError('Input matrix is not convertible to numpy array!')

        if matrix.ndim != 2 or not_symmetric(matrix):
            raise ValueError('Input matrix appears to be NOT 2D or symmetric! '
                             'Symmetry is needed for a valid kernel.')

        self._KM = matrix
        self.num_samples = self._KM.shape[0]

        if name is None:
            self.name = 'Precomputed'
        else:
            self.name = str(name)


    def __len__(self):
        """size of kernel matrix"""

        return self.size


    @property
    def size(self):
        """size of kernel matrix"""

        return self._KM.shape[0]


    @property
    def full(self):
        """Returns the full kernel matrix (in dense format, as its already precomputed)"""
        return self._KM


    @property
    def diag(self):
        """Returns the diagonal of the kernel matrix"""

        return self._KM.diagonal()


    def __getitem__(self, index_obj):
        """Access the matrix"""

        try:
            return self._KM[index_obj]
        except:
            raise KMAccessError('Invalid attempt to access the 2D kernel matrix!')


    def __str__(self):
        """human readable presentation"""

        return "{}(num_samples={})".format(self.name, self.num_samples)


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


class ConstantKernelMatrix(object):
    """Custom KM to represent a constant """


    def __init__(self, num_samples=None, value=0.0, name=None, dtype='float'):

        self.num_samples = num_samples
        self.const_value = value
        self.dtype = dtype

        if name is None:
            self.name = 'Constant'
        else:
            self.name = str(name)


    def __len__(self):
        """size of kernel matrix"""

        return self.size


    @property
    def size(self):
        """size of kernel matrix"""
        return self.num_samples


    @property
    def full(self):
        """Returns the full kernel matrix (in dense format)"""

        if not hasattr(self, '_KM'):
            self._KM = np.full((self.num_samples, self.num_samples),
                               fill_value=self.const_value, dtype=self.dtype)

        return self._KM


    @property
    def diag(self):
        """Returns the diagonal of the kernel matrix"""

        return np.full((self.num_samples,),
                       fill_value=self.const_value, dtype=self.dtype)


    def __getitem__(self, index_obj):
        """Access the matrix"""

        try:
            return self._KM[index_obj]
        except:
            raise KMAccessError('Invalid attempt to access the 2D kernel matrix!')


    def __str__(self):
        """human readable presentation"""

        return "{}(value={},num_samples={})".format(self.name, self.const_value,
                                                    self.num_samples)


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


VALID_KERNEL_MATRIX_TYPES = (KernelMatrix, KernelMatrixPrecomputed, np.ndarray)


class KernelSet(object):
    """Container class to manage a set of KernelMatrix instances."""


    def __init__(self, km_set, name='KernelSet'):
        """Constructor."""

        # to denote no KM has been added yet
        self._is_init = False
        self.name = name

        # type must be Sequence (not just Iterable)
        #   as we need to index it from 1 (second element in the Iterable)
        if isinstance(km_set, Sequence):
            # adding the first kernel by itself
            # needed for compatibility tests
            self._initialize(km_set[0])

            # add the remaining if they are compatible
            for km in km_set[1:]:
                self.append(km)

        elif isinstance(km_set, VALID_KERNEL_MATRIX_TYPES):
            self._initialize(km_set)
        else:
            raise TypeError('Unknown type of input matrix! Must be one of:\n'
                            '{}'.format(VALID_KERNEL_MATRIX_TYPES))


    def _initialize(self, KM):
        """Method to initialize and set key compatibility parameters"""

        if not self._is_init:
            self._km_set = list()

            if isinstance(KM, (KernelMatrix, KernelMatrixPrecomputed)):
                self._km_set.append(KM)
                self._num_samples = KM.size
            elif isinstance(KM, np.ndarray):
                self._km_set.append(KernelMatrixPrecomputed(KM))
                self._num_samples = KM.shape[0]

            self._is_init = True


    @property
    def size(self):
        """Number of kernel matrices in this set"""

        return len(self._km_set)


    @property
    def num_samples(self):
        """Number of samples in each individual kernel matrix """

        return self._num_samples


    def __len__(self):
        """Returns the number of kernels in this set"""

        return len(self._km_set)


    def append(self, KM):
        """
        Method to add a new kernel to the set.

        Checks to ensure the new KM is compatible in size to the existing set.
        """

        if not isinstance(KM, (BaseKernelFunction, KernelMatrix,
                               KernelMatrixPrecomputed)):
            KM = KernelMatrixPrecomputed(KM)

        if self._num_samples != KM.num_samples:
            raise KMSetAdditionError('Dimension of this KM {} is incompatible '
                                     'with KMSet of {}! '
                                     ''.format(KM.num_samples, self.num_samples))

        self._km_set.append(KM)


    def __getitem__(self, index):
        """To retrieve individual kernels"""

        if not isinstance(index, int):
            raise ValueError('Only integer indices are permitted, '
                             'accessing one KM at a time')

        if index < 0 or index >= self.size:
            raise IndexError('Index out of range for KernelSet of size {}'
                             ''.format(self.size))

        return self._km_set[index]


    def __str__(self):
        """Human readable repr"""

        return "{}({} kernels, {} samples):\n\t{} " \
               "".format(self.name, self.size, self.num_samples,
                         "\n\t".join(map(str, self._km_set)))

    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__

    def __iter__(self):
        """Making an iterable."""

        for index in range(self.size):
            yield self._km_set[index]


    def attach_to(self, sample):
        """Attach this kernel to a given sample.

        Any previous evaluations to other samples and their results will be reset.

        Parameters
        ----------
        sample : ndarray
            Input sample to operate on
            Must be 2D of shape (num_samples, num_features)

        """

        self.sample = ensure_ndarray_2D(sample)
        if self._num_samples is not None and sample.shape[0] != self._num_samples:
            raise ValueError('Number of samples in input differ from this KernelSet')
        else:
            self._num_samples = sample.shape[0]

        for index in range(self.size):
            self._km_set[index].attach_to(sample)


    def extend(self, another_km_set):
        """Combines two sets into one"""

        if not isinstance(another_km_set, KernelSet):
            raise KMSetAdditionError('Input is not a KernelSet!'
                                     'Build a KernelSet() first.')

        if another_km_set.num_samples != self.num_samples:
            raise ValueError('The two KernelSets are not compatible (in size: # samples)')

        for km in another_km_set:
            self.append(km)

